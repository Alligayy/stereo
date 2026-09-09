import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time as t
from pycuda.autoinit import context
import os
from collections import Counter
from numba import jit
code_dir = os.path.dirname(os.path.realpath(__file__))

@jit
def subfuction(h,w,result,disimg,lrc,Ts,Th,maxdis):
    count=0;val=np.zeros(shape=maxdis,dtype=np.uint8)
    for y in range(h-result[h,w,2],h+result[h,w,3]):
        for x in range(w-result[y,w,0],w+result[y,w,1]):
            if lrc[y,x]==255:
                val[disimg[y,x]]=val[disimg[y,x]]+1
                count=count+1
    if count<=Ts:
        return disimg[h,w],lrc[h,w]
    else:
        time=np.max(val)
        if time/count >Th:
            return np.argmax(val),255
        else:
            return disimg[h,w],lrc[h,w]
    
@jit
def Iterative_Region_Voting(disimg,lrc,result,Ts,Th,maxdis):
    lrca=lrc.copy()
    for h in range(lrc.shape[0]):
        for w in range(lrc.shape[1]):
            if lrc[h,w]!=255:
                disimg[h,w],lrca[h,w]=subfuction(h,w,result,disimg,lrc,Ts,Th,maxdis)
    return disimg,lrca

@jit                      
def Sub_pixel_Enhancement(disimg, costVolume, MaxDis):
    disimg=disimg.astype(np.float32)
    for h in range(0,disimg.shape[0]):
        for w in range(0,disimg.shape[1]):
            d0=int(disimg[h,w]);d1=d0-1;d2=d0+1
            if (costVolume[h,w,d2]+costVolume[h,w,d1]-2*costVolume[h,w,d0])!=0 and d1>=0 and d2<MaxDis:
                disimg[h,w]=disimg[h,w]-((costVolume[h,w,d2]-costVolume[h,w,d1])/\
                                        (2*(costVolume[h,w,d2]+costVolume[h,w,d1]-2*costVolume[h,w,d0])))
    return disimg

class ADCensusModel:
    def __init__(self, module_path=os.path.join(code_dir, 'adcensus_core/module.cpp'), **kwargs):
        with open(module_path, 'r', encoding='utf-8') as file:
            module_code = file.read()

        self.mod = SourceModule(module_code)

        self.tracker= cv2.legacy.TrackerMedianFlow_create()
        self.Census=self.mod.get_function("Census")
        self.AD=self.mod.get_function("AD")
        self.ComputCost=self.mod.get_function("ComputCost")
        self.get_arm=self.mod.get_function("get_arm")
        self.AggH=self.mod.get_function("AggH")
        self.AggV=self.mod.get_function("AggV")
        self.LRCheck=self.mod.get_function("LRCheck")
        self.Postprocess=self.mod.get_function("Postprocess")
        self.select_disparity=self.mod.get_function("select_disparity")

        self.MaxDis = 80
        self.H, self.W = -1, -1

        self.N = self.H * self.W * self.MaxDis
        self.NN = self.H * self.W

        self.T1 = 23
        self.T2 = 11
        self.L1 = 40
        self.L2 = 20

    def __memory_init(self, H, W):
        self.left_census = drv.mem_alloc(H*W*self.MaxDis*4)
        self.right_census = drv.mem_alloc(H*W*self.MaxDis*4)
        self.left_ad = drv.mem_alloc(H*W*self.MaxDis*4)
        self.right_ad = drv.mem_alloc(H*W*self.MaxDis*4)
        self.left_cost = drv.mem_alloc(H*W*self.MaxDis*4)
        self.right_cost = drv.mem_alloc(H*W*self.MaxDis*4)
        self.resultL = drv.mem_alloc(H*W*4*4)
        self.resultR = drv.mem_alloc(H*W*4*4)
        self.left_dis = drv.mem_alloc(H*W*4)
        self.right_dis = drv.mem_alloc(H*W*4)
        self.final = drv.mem_alloc(H*W*4)
        self.finalnew = drv.mem_alloc(H*W*4)
        self.imCLdata = drv.mem_alloc(H*W*3*4)
        self.imCRdata = drv.mem_alloc(H*W*3*4)
        self.leftdis = np.zeros(shape=(H,W),dtype=np.int32)
        self.resultLL = np.zeros(shape=(H,W,4),dtype=np.int32)
        self.lrc = np.zeros(shape=(H,W),dtype=np.int32)
        self.left_agged = np.zeros(shape=(H,W,self.MaxDis),dtype=np.int32)

        self.H, self.W = H, W

        self.N = self.H * self.W * self.MaxDis
        self.NN = self.H * self.W

        

    def get_disparity(self, imCL, imCR):
        if len(imCL.shape) == 3:
            imL = cv2.cvtColor(imCL,cv2.COLOR_BGR2GRAY)
            imR = cv2.cvtColor(imCR,cv2.COLOR_BGR2GRAY)
        else:
            imL, imR = imCL.copy(), imCR.copy()
            imCL = cv2.cvtColor(imCL, cv2.COLOR_GRAY2BGR)
            imCR = cv2.cvtColor(imCR, cv2.COLOR_GRAY2BGR)


        H, W = imL.shape[:2]
        if self.H != H or self.W != W:
            self.__memory_init(H, W)
          
        # imCLd = imCL.reshape(H*W*3).astype(np.float32)
        # imCRd = imCR.reshape(H*W*3).astype(np.float32)
        drv.memcpy_htod(self.imCLdata,imCL)
        drv.memcpy_htod(self.imCRdata,imCR)
        imLdata = imL.astype(np.float32)
        imRdata = imR.astype(np.float32)
        self.Census(self.left_census, self.right_census,drv.In(imLdata),drv.In(imRdata),np.uint32(H),np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AD(self.imCLdata,self.imCRdata,self.left_ad,self.right_ad,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.ComputCost(self.left_census, self.left_ad, self.left_cost,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.ComputCost(self.right_census, self.right_ad, self.right_cost,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.get_arm(self.imCLdata,self.resultL,np.uint32(self.T1),np.uint32(self.T2),np.uint32(self.L1),np.uint32(self.L2),np.uint32(W),np.uint32(H),np.uint32(self.NN),block=(1024,1,1),grid=(4800,1,1))
        self.get_arm(self.imCRdata,self.resultR,np.uint32(self.T1),np.uint32(self.T2),np.uint32(self.L1),np.uint32(self.L2),np.uint32(W),np.uint32(H),np.uint32(self.NN),block=(1024,1,1),grid=(4800,1,1))
        self.AggH(self.left_cost,self.left_census,self.resultL,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggV(self.left_census,self.left_cost,self.resultL,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggH(self.left_cost,self.left_census,self.resultL,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggV(self.left_census,self.left_cost,self.resultL,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggH(self.right_cost,self.right_census,self.resultR,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggV(self.right_census,self.right_cost,self.resultR,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggH(self.right_cost,self.right_census,self.resultR,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.AggV(self.right_census,self.right_cost,self.resultR,np.uint32(W),np.uint32(self.N),np.uint32(self.MaxDis),block=(self.MaxDis,1,1),grid=(H,W,1))
        self.select_disparity(self.left_cost,self.left_dis,np.uint32(W),np.uint32(self.NN),np.uint32(self.MaxDis),block=(1024,1,1),grid=(4800,1,1))
        self.select_disparity(self.right_cost,self.right_dis,np.uint32(W),np.uint32(self.NN),np.uint32(self.MaxDis),block=(1024,1,1),grid=(4800,1,1))
        self.LRCheck(self.left_dis,self.right_dis,self.final,np.uint32(W),np.uint32(self.NN),block=(1024,1,1),grid=(4800,1,1))
        context.synchronize()
        drv.memcpy_dtoh(self.leftdis,self.left_dis)
        drv.memcpy_dtoh(self.resultLL,self.resultL)
        drv.memcpy_dtoh(self.lrc,self.final)
        drv.memcpy_dtoh(self.left_agged,self.left_cost)
        for _ in range(5):
            self.leftdis, self.lrc=Iterative_Region_Voting(self.leftdis,self.lrc,self.resultLL,20,0.4,self.MaxDis)

        self.Postprocess(drv.InOut(self.lrc),drv.InOut(self.leftdis),np.uint32(W),np.uint32(H),np.uint32(self.NN),block=(1024,1,1),grid=(4800,1,1))
        ccdd = Sub_pixel_Enhancement(self.leftdis, self.left_agged, self.MaxDis)
        ccdd = cv2.medianBlur(ccdd, 3)

        disp = ccdd.astype(np.float32)

        return disp
    

if __name__ == '__main__':
    ad = ADCensusModel()