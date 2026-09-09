#include <stdio.h> 
__global__ void Census(float *censusL,float *censusR,float *imL,float *imR,int H,int W,int N,int maxdis){


    int d=threadIdx.x;
    int h=blockIdx.x;
    int w=blockIdx.y;


    float count=0.0;


    if((w-d)>=4 && h>=3 && h<H-3 && w<W-4){
        for(int x=h-3;x<=h+3;x++){
            for(int y=w-4;y<=w+4;y++){
                if((imL[x*W+y] < imL[h*W+w]) != (imR[x*W+y-d] < imR[h*W+w-d])){
                count++;
                }
            }
        }
        censusL[h*W*maxdis+w*maxdis+d]=count;
        censusR[h*W*maxdis+(w-d)*maxdis+d]=count;
    }
    else{
        censusL[h*W*maxdis+w*maxdis+d]=64.0;
    }

    if((w+d)>=W-4 || h<3 || h>=H-3 || w<4){
        censusR[h*W*maxdis+w*maxdis+d]=64.0;
    }

}




__global__ void AD(float *imL,float *imR,float *outL,float *outR,int W,int N,int maxdis){


    int d=threadIdx.x;
    int h=blockIdx.x;
    int w=blockIdx.y;
    if(w-d>=0){
        outL[h*W*maxdis+w*maxdis+d]=(fabs(imL[h*W*3+w*3]-imR[h*W*3+(w-d)*3])+fabs(imL[h*W*3+w*3+1]-imR[h*W*3+(w-d)*3+1])+fabs(imL[h*W*3+w*3+2]-imR[h*W*3+(w-d)*3+2]))/3.0;
        outR[h*W*maxdis+(w-d)*maxdis+d]=outL[h*W*maxdis+w*maxdis+d];
    }
    else{
        outL[h*W*maxdis+w*maxdis+d]=255.0;
    }
    if(w+d>=W){
        outR[h*W*maxdis+w*maxdis+d]=255.0;
    }
}




__global__ void ComputCost(float *ce,float *ad,float *out,int W,int N,int maxdis){

    int d=threadIdx.x;
    int h=blockIdx.x;
    int w=blockIdx.y;
    int idx=h*W*maxdis+w*maxdis+d;
    float lameta1=30.0;
    float lameta2=10.0;
    out[idx]=2.0-exp(-1*ce[idx]/lameta1)-exp(-1*ad[idx]/lameta2);

}


__global__ void get_arm(float *data,int *result,int T1,int T2,int L1,int L2,int W,int H,int N){
    int idx=threadIdx.x + blockIdx.x * blockDim.x;
    float Colordiff,Spidiff,Neidiff;
    if(idx<N){
        int h=idx/W;
        int w=idx%W;
        for(int l=w-1;l>=0;l--){
            Colordiff=max(abs(data[h*W*3+w*3]-data[h*W*3+l*3]),abs(data[h*W*3+w*3+1]-data[h*W*3+l*3+1]));
            Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[h*W*3+l*3+2]));
            Spidiff=w-l;
            Neidiff=max(abs(data[h*W*3+(l+1)*3]-data[h*W*3+l*3]),abs(data[h*W*3+(l+1)*3+1]-data[h*W*3+l*3+1]));
            Neidiff=max(Neidiff,abs(data[h*W*3+(l+1)*3+2]-data[h*W*3+l*3+2]));
            if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && l!=0)
                continue;
            if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && l!=0)
                continue;
            result[h*W*4+w*4]=max(int(Spidiff-1),1);
            break;
        }

        for(int r=w+1;r<W;r++)
        {
            Colordiff=max(abs(data[h*W*3+w*3]-data[h*W*3+r*3]),abs(data[h*W*3+w*3+1]-data[h*W*3+r*3+1]));
            Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[h*W*3+r*3+2]));
            Spidiff=r-w;
            Neidiff=max(abs(data[h*W*3+(r-1)*3]-data[h*W*3+r*3]),abs(data[h*W*3+(r-1)*3+1]-data[h*W*3+r*3+1]));
            Neidiff=max(Neidiff,abs(data[h*W*3+(r-1)*3+2]-data[h*W*3+r*3+2]));
            if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && r!=W-1)
                continue;
            if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && r!=W-1)
                continue;
            result[h*W*4+w*4+1]=max(int(Spidiff),2);
            break;
        }


        for(int t=h-1;t>=0;t--){
            Colordiff=max(abs(data[h*W*3+w*3]-data[t*W*3+w*3]),abs(data[h*W*3+w*3+1]-data[t*W*3+w*3+1]));
            Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[t*W*3+w*3+2]));
            Spidiff=h-t;
            Neidiff=max(abs(data[t*W*3+w*3]-data[(t+1)*W*3+w*3]),abs(data[t*W*3+w*3+1]-data[(t+1)*W*3+w*3+1]));
            Neidiff=max(Neidiff,abs(data[t*W*3+w*3+2]-data[(t+1)*W*3+w*3+2]));
            if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && t!=0)
                continue;
            if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && t!=0)
                continue;
            result[h*W*4+w*4+2]=max(int(Spidiff-1),1);
            break;
        }

        for(int b=h+1;b<H;b++){
            Colordiff=max(abs(data[h*W*3+w*3]-data[b*W*3+w*3]),abs(data[h*W*3+w*3+1]-data[b*W*3+w*3+1]));
            Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[b*W*3+w*3+2]));
            Spidiff=b-h;
            Neidiff=max(abs(data[b*W*3+w*3]-data[(b-1)*W*3+w*3]),abs(data[b*W*3+w*3+1]-data[(b-1)*W*3+w*3+1]));
            Neidiff=max(Neidiff,abs(data[b*W*3+w*3+2]-data[(b-1)*W*3+w*3+2]));
            if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && b!=H-1)
                continue;
            if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && b!=H-1)
                continue;
            result[h*W*4+w*4+3]=max(int(Spidiff),2);
            break;
        }
    }
}



__global__ void AggH(float *costvolume,float *out,int *result,int W,int N,int maxdis){

    int d=threadIdx.x;
    int h=blockIdx.x;
    int w=blockIdx.y;
    int idx=h*W*maxdis+w*maxdis+d;
    out[idx]=0.0;
    for(int l=w-result[h*W*4+w*4];l<w+result[h*W*4+w*4+1];l++){
        out[idx]=out[idx]+costvolume[h*W*maxdis+l*maxdis+d];
    }
}


__global__ void AggV(float *costvolume,float *out,int *result,int W,int N,int maxdis){

    int d=threadIdx.x;
    int h=blockIdx.x;
    int w=blockIdx.y;
    int idx=h*W*maxdis+w*maxdis+d;
    out[idx]=0.0;
    for(int l=h-result[h*W*4+w*4+2];l<h+result[h*W*4+w*4+3];l++)
    {
        out[idx]=out[idx]+costvolume[l*W*maxdis+w*maxdis+d];
    }
}




__global__ void LRCheck(int *left_dis,int *right_dis,int *out,int W,int N,int maxdis){

    int idx=threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<N){
        int h=idx/W;
        int w=idx%W;
        int d=left_dis[h*W+w];
        int flag=0;
        if(w-d>=0){
            if (abs(left_dis[h*W+w]-right_dis[h*W+w-d])>1){
                for(int x=0;x<maxdis;x++){
                    if(w-x>=0){
                        if(right_dis[h*W+w-x] > left_dis[h*W+w])
                            flag=1;
                        break;
                    }
                }
                if(flag==1) out[h*W+w]=125;
                else out[h*W+w]=0;
                

            }
            else{
                out[h*W+w]=255;
            }
        }
        else{
            out[idx]=0;
        }
    }
}


__global__ void Postprocess(int *ref,int *disimg,int W,int H,int N){
    int idx=threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<N){
        int h=idx/W;
        int w=idx%W;
        int ah=h;
        int aw=w;
        int index=0;
        int val[8];
        int temp;
        int symbol=0;
        if(ref[h*W+w]==125 || ref[h*W+w]==0){
            for(int x=-1;x<=1;x++){
                for(int y=-1;y<=1;y++){
                    if(x==0&&y==0)
                        continue;
                    while(ah+x>=0 && ah+x<H && aw+y>=0 && aw+y<W && ref[(ah+x)*W+(aw+y)] !=255){
                        ah=ah+x;
                        aw=aw+y;
                    }
                    ah=ah+x;
                    aw=aw+y;
                    if(ah>0 && ah<H && aw>0 && aw<W){
                        val[index]=disimg[ah*W+aw];
                        ++index;
                    }
                    ah=h;
                    aw=w;
                }
            }

            for (int i=0; i<index-1; ++i){
                for (int j=0; j<index-1-i; ++j){
                    if (val[j] < val[j+1]){
                        temp = val[j];
                        val[j] = val[j+1];
                        val[j+1] = temp;
                    }
                }
            }
        }

        if(ref[h*W+w]==0){
            for(int x=-1;x<=1;x++){
                for(int y=-1;y<=1;y++){
                    if(x==0&&y==0)
                        continue;
                    if(ah+x>=0 && ah+x<H && aw+y>=0 && aw+y<W){
                        ah=ah+x;
                        aw=aw+y;
                        if(ref[ah*W+aw]==125){
                            symbol=1;
                        }
                        ah=h;
                        aw=w;
                    }
                }
            }
            if(symbol==1){
                disimg[h*W+w]=val[index-2];
            }
            else{
                if(index%2==1)
                    disimg[h*W+w]=val[int(index/2)];
                else
                    disimg[h*W+w]=int((val[int(index/2)-1]+val[int(index/2)])/2);
            }
        }

        if(ref[h*W+w]==125){
            disimg[h*W+w]=val[index-2];
        }

            
    }
}


__global__ void Iterative_Region_Voting(int *disimg,int *result,int *ref,int *refnew,int W,int N,int maxdis){
    int idx=threadIdx.x + blockIdx.x*blockDim.x;
    if(idx<N){
        int h=idx/W;
        int w=idx%W;
        int val[256];
        for(int j=0;j<256;j++)
            val[j]=0;
        int count=0;
        int newval=0;
        int temp=0;
        int Ts=20;
        float Th=0.4;
        refnew[h*W+w]=ref[h*W+w];

        if(ref[h*W+w] != 255){
            for(int t=h-result[h*W*4+w*4+2];t<h+result[h*W*4+w*4+3];t++){
                for(int l=w-result[t*W*4+w*4];l<w+result[t*W*4+w*4+1];l++){
                    if(ref[t*W+l]==255){
                        val[disimg[t*W+l]]++;
                        count++;
                    }
                }
            }


            for(int i=0;i<maxdis;i++){
                if(val[i]>temp){
                    newval=i;
                    temp=val[i];
                } 
            }
            __syncthreads();


            if(count > Ts){
                if(float(temp/count)>Th){
                    refnew[h*W+w]=255;
                    disimg[h*W+w]=newval;
                }
            }
        }
    } 
}

__global__ void select_disparity(float *cost,int *out,int W,int N,int maxdis){
int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx<N){
    int h=idx/W;
    int w=idx%W;
    int index=0;
    int min=cost[h*W*maxdis+w*maxdis+index];
    for(int d=0;d<maxdis;d++){
        if(cost[h*W*maxdis+w*maxdis+d]<min){
            index=d;
            min=cost[h*W*maxdis+w*maxdis+d];
        } 
    }
    out[h*W+w]=index;
    }
}




