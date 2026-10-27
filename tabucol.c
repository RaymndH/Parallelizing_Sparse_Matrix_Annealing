// TabuCol (Hertz & de Werra 1987) for q=5 colouring: sequential tabu search on the number of conflicting edges.
// usage: ./tabucol graph max_iters seeds
//   Each iteration picks the best (node in conflict, new colour) move by the neighbour-colour table gamma[v][c],
//   skipping tabu moves unless they beat the best conflict count so far (aspiration). After moving v from o to n,
//   the move (v,o) is tabu for  rand(0..9) + 0.6 * (#conflicting nodes)  iterations. Ties are broken at random.
// build: gcc -O3 -march=native -o tabucol tabucol.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define Q 5
static uint64_t s; static inline uint64_t rnd(){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"r"); int N; char line[256]; if(!fgets(line,sizeof line,f))return 1;
  if(sscanf(line,"N = %d",&N)!=1 && sscanf(line,"%d",&N)!=1){fprintf(stderr,"bad header\n");return 1;}
  int M=0,cap=1<<21,*ea=malloc(cap*4),*eb=malloc(cap*4),a,b;
  while(fscanf(f,"%d %d",&a,&b)==2){ if(M==cap){cap*=2;ea=realloc(ea,cap*4);eb=realloc(eb,cap*4);} ea[M]=a;eb[M]=b;M++;}
  int*deg=calloc(N+1,4),*ptr=calloc(N+1,4),*nbr=malloc(2*(size_t)M*4),*fill=calloc(N,4);
  for(int i=0;i<M;i++){deg[ea[i]]++;deg[eb[i]]++;}
  for(int i=0;i<N;i++)ptr[i+1]=ptr[i]+deg[i];
  for(int i=0;i<M;i++){nbr[ptr[ea[i]]+fill[ea[i]]++]=eb[i];nbr[ptr[eb[i]]+fill[eb[i]]++]=ea[i];}
  long maxit=atol(argv[2]); int seeds=atoi(argv[3]);
  setvbuf(stdout,NULL,_IOLBF,0);
  int*q=malloc(N*4),*g=malloc((size_t)N*Q*4),*cl=malloc(N*4),*pos=malloc(N*4); long*tabu=malloc((size_t)N*Q*8);
  int ok=0; double tsum=0,fsum=0;
  for(int sd=0;sd<seeds;sd++){
    s=(0x9E3779B97F4A7C15ULL*(sd+1))|1; for(int i=0;i<20;i++)rnd();
    for(int i=0;i<N;i++)q[i]=rnd()%Q;
    for(int i=0;i<N*Q;i++){g[i]=0;tabu[i]=0;}
    for(int i=0;i<N;i++)for(int k=ptr[i];k<ptr[i+1];k++)g[i*Q+q[nbr[k]]]++;
    int ncl=0; long E=0;
    for(int i=0;i<N;i++){ pos[i]=-1; if(g[i*Q+q[i]]>0){pos[i]=ncl;cl[ncl++]=i;} E+=g[i*Q+q[i]]; } E/=2;
    long best=E,it; clock_t t0=clock();
    for(it=0;it<maxit&&E>0;it++){
      int bv=-1,bc=-1,nb=0; int bd=1<<30;
      for(int j=0;j<ncl;j++){ int v=cl[j],o=q[v],go=g[v*Q+o];
        for(int c=0;c<Q;c++){ if(c==o)continue; int d=g[v*Q+c]-go;
          if(tabu[(size_t)v*Q+c]>it && E+d>=best)continue;   // tabu unless it gives a new best
          if(d<bd){bd=d;bv=v;bc=c;nb=1;} else if(d==bd && rnd()%(++nb)==0){bv=v;bc=c;} } }
      if(bv<0){ bv=cl[rnd()%ncl]; bc=(q[bv]+1+rnd()%(Q-1))%Q; bd=g[bv*Q+bc]-g[bv*Q+q[bv]]; }   // everything tabu: random move
      int v=bv,o=q[v],n=bc; q[v]=n; E+=bd;
      tabu[(size_t)v*Q+o]=it+rnd()%10+(long)(0.6*ncl);
      for(int k=ptr[v];k<ptr[v+1];k++){ int u=nbr[k]; g[u*Q+o]--; g[u*Q+n]++;
        int inc=g[u*Q+q[u]]>0;
        if(inc&&pos[u]<0){pos[u]=ncl;cl[ncl++]=u;}
        else if(!inc&&pos[u]>=0){ int w=cl[--ncl]; cl[pos[u]]=w; pos[w]=pos[u]; pos[u]=-1; } }
      int inc=g[v*Q+q[v]]>0;
      if(inc&&pos[v]<0){pos[v]=ncl;cl[ncl++]=v;}
      else if(!inc&&pos[v]>=0){ int w=cl[--ncl]; cl[pos[v]]=w; pos[w]=pos[v]; pos[v]=-1; }
      if(E<best)best=E;
    }
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("seed=%d conflicts=%ld best=%ld iters=%ld time=%.3fs (%.2f us/iter)\n",sd,E,best,it,el,it?1e6*el/it:0.0);
    if(E==0){ok++;tsum+=el;} fsum+=E;
  }
  printf("success %d/%d, mean time to solve %.3fs, mean final conflicts %.1f\n",ok,seeds,ok?tsum/ok:0,fsum/seeds);
}
