// Sequential single-spin Metropolis annealing for q=5 coloring (baseline for anneal_gpu.py).
// usage: ./seq graph T0 T1 sweeps seeds [splits.txt]
//   no splits file: random node order every sweep
//   splits file (10 lines of N class labels): each sweep visits nodes set by set from a random split,
//     which gives exactly the same result as updating each independent set in parallel
//   env LIN=1: linear schedule T0->T1 (like testcode.cpp), otherwise geometric
// build: gcc -O3 -march=native -o seq seq.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
static uint64_t s; static inline uint64_t rnd(){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
static inline double urand(){ return (rnd()>>11)*(1.0/9007199254740992.0); }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"r"); int N,Q=5; char line[256]; if(!fgets(line,sizeof line,f))return 1;
  if(sscanf(line,"N = %d",&N)!=1 && sscanf(line,"%d",&N)!=1){fprintf(stderr,"bad header\n");return 1;}
  int M=0,cap=1<<21,*ea=malloc(cap*4),*eb=malloc(cap*4),a,b;
  while(fscanf(f,"%d %d",&a,&b)==2){ea[M]=a;eb[M]=b;M++;}
  int*deg=calloc(N+1,4),*ptr=calloc(N+1,4),*nbr=malloc(2*M*4),*fill=calloc(N,4);
  for(int i=0;i<M;i++){deg[ea[i]]++;deg[eb[i]]++;}
  for(int i=0;i<N;i++)ptr[i+1]=ptr[i]+deg[i];
  for(int i=0;i<M;i++){nbr[ptr[ea[i]]+fill[ea[i]]++]=eb[i];nbr[ptr[eb[i]]+fill[eb[i]]++]=ea[i];}
  double T0=atof(argv[2]),T1=atof(argv[3]); int sweeps=atoi(argv[4]),seeds=atoi(argv[5]);
  int*q=malloc(N*4),*A=malloc(N*Q*4),*perm=malloc(N*4);
  // optional: splits file (10 lines x N class labels) -> group order
  int NS=0,*gorder=NULL;
  if(argc>6){FILE*g=fopen(argv[6],"r"); NS=10; gorder=malloc((size_t)NS*N*4); int*lab=malloc(N*4);
    for(int sp=0;sp<NS;sp++){int K=0; for(int i=0;i<N;i++){if(fscanf(g,"%d",&lab[i])!=1)return 1; if(lab[i]+1>K)K=lab[i]+1;}
      int w=0; for(int c=0;c<K;c++)for(int i=0;i<N;i++)if(lab[i]==c)gorder[(size_t)sp*N+w++]=i;}
    printf("mode: group order (%d splits)\n",NS);} else printf("mode: random order\n"); int ok=0; double tsum=0; long swsum=0;
  for(int sd=0;sd<seeds;sd++){
    s=0x9E3779B97F4A7C15ULL*(sd+1);
    for(int i=0;i<N;i++)q[i]=rnd()%Q;
    for(int i=0;i<N*Q;i++)A[i]=0;
    for(int i=0;i<N;i++)for(int k=ptr[i];k<ptr[i+1];k++)A[i*Q+q[nbr[k]]]++;
    long E=0; for(int i=0;i<M;i++)E+=q[ea[i]]==q[eb[i]];
    for(int i=0;i<N;i++)perm[i]=i;
    clock_t t=clock(); int used=sweeps;
    for(int sw=0;sw<sweeps;sw++){
      double T=getenv("LIN")?T0-(T0-T1)*(double)sw/(sweeps-1):T0*pow(T1/T0,(double)sw/(sweeps-1));
      int*ord=perm;
      if(NS) ord=gorder+(size_t)(rnd()%NS)*N;
      else for(int i=N-1;i>0;i--){int j=rnd()%(i+1),x=perm[i];perm[i]=perm[j];perm[j]=x;}
      for(int ii=0;ii<N;ii++){int v=ord[ii],o=q[v],n=(o+1+rnd()%(Q-1))%Q;
        int dE=A[v*Q+n]-A[v*Q+o];
        if(dE<=0||urand()<exp(-dE/T)){q[v]=n;E+=dE;
          for(int k=ptr[v];k<ptr[v+1];k++){A[nbr[k]*Q+o]--;A[nbr[k]*Q+n]++;}}}
      if(E==0){used=sw+1;break;}
    }
    double el=(double)(clock()-t)/CLOCKS_PER_SEC;
    printf("seed=%d conflicts=%ld sweeps=%d time=%.3fs (%.1f us/sweep)\n",sd,E,used,el,1e6*el/used);
    if(E==0){ok++;swsum+=used;tsum+=el;}
  }
  printf("success %d/%d, mean sweeps to solve %.0f, mean time %.3fs\n",ok,seeds,ok?(double)swsum/ok:0,ok?tsum/ok:0);
}
