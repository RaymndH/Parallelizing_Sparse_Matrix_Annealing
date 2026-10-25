// OpenMP version of seq.c: independent-set parallel Metropolis annealing for q=5 coloring on the CPU.
// usage: OMP_NUM_THREADS=16 ./par graph T0 T1 sweeps seeds
//   10 greedy splits (random-order first-fit coloring) are built at startup; each sweep picks one at random
//   and updates each independent set in parallel (set after set), matching the GPU scheme.
//   Random numbers are a hash of (seed, sweep, node), so results do not depend on the thread count.
//   env LIN=1: linear schedule T0->T1, otherwise geometric
//   env TRACE=k: print conflicts every k sweeps
// build: gcc -O3 -march=native -fopenmp -o par par.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#define Q 5
#define NS 10
static inline uint64_t mix(uint64_t z){ z+=0x9E3779B97F4A7C15ULL; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
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
  double T0=atof(argv[2]),T1=atof(argv[3]); int sweeps=atoi(argv[4]),seeds=atoi(argv[5]);
  int lin=getenv("LIN")!=NULL, trace=getenv("TRACE")?atoi(getenv("TRACE")):0;
  setvbuf(stdout,NULL,_IOLBF,0);
  // splits: order[sp] lists nodes class by class, start[sp][c] gives class offsets
  int*order=malloc((size_t)NS*N*4),*start[NS],K[NS];
  s=88172645463325252ULL;
  { int*perm=malloc(N*4),*col=malloc(N*4),*cnt=malloc((size_t)(N+1)*4);
    for(int sp=0;sp<NS;sp++){
      for(int i=0;i<N;i++)perm[i]=i;
      for(int i=N-1;i>0;i--){int j=rnd()%(i+1),x=perm[i];perm[i]=perm[j];perm[j]=x;}
      for(int i=0;i<N;i++)col[i]=-1;
      int k=0;
      for(int ii=0;ii<N;ii++){int v=perm[ii]; unsigned long long used=0;
        for(int e=ptr[v];e<ptr[v+1];e++)if(col[nbr[e]]>=0&&col[nbr[e]]<64)used|=1ULL<<col[nbr[e]];
        int c=0; while(used>>c&1)c++; col[v]=c; if(c+1>k)k=c+1;}
      K[sp]=k; start[sp]=calloc(k+1,4);
      for(int i=0;i<N;i++)start[sp][col[i]+1]++;
      for(int c=0;c<k;c++)start[sp][c+1]+=start[sp][c];
      for(int c=0;c<k;c++)cnt[c]=start[sp][c];
      for(int i=0;i<N;i++)order[(size_t)sp*N+cnt[col[i]]++]=i;
    }
    printf("threads=%d  sets/split=",omp_get_max_threads()); for(int sp=0;sp<NS;sp++)printf("%d ",K[sp]); printf("\n");
  }
  int*q=malloc(N*4),*A=malloc((size_t)N*Q*4);
  int ok=0; double tsum=0; long swsum=0;
  for(int sd=0;sd<seeds;sd++){
    uint64_t seed=mix(0x1234567ULL+sd); s=seed|1;
    for(int i=0;i<N;i++)q[i]=rnd()%Q;
    for(int i=0;i<N*Q;i++)A[i]=0;
    for(int i=0;i<N;i++)for(int k=ptr[i];k<ptr[i+1];k++)A[i*Q+q[nbr[k]]]++;
    long E=0; for(int i=0;i<M;i++)E+=q[ea[i]]==q[eb[i]];
    long dEs=0; int stop=0,used=sweeps,sp=0; double T=T0;
    double t=omp_get_wtime();
    #pragma omp parallel
    for(int sw=0;sw<sweeps;sw++){
      #pragma omp single
      { T=lin?T0-(T0-T1)*(double)sw/(sweeps-1):T0*pow(T1/T0,(double)sw/(sweeps-1)); sp=mix(seed^(uint64_t)sw)%NS; }
      for(int c=0;c<K[sp];c++){
        const int*ord=order+(size_t)sp*N+start[sp][c]; int n_c=start[sp][c+1]-start[sp][c];
        #pragma omp for schedule(static) reduction(+:dEs)
        for(int ii=0;ii<n_c;ii++){
          int v=ord[ii],o=q[v]; uint64_t h=mix(seed^((uint64_t)sw*N+v)*0x9E3779B97F4A7C15ULL);
          int n=(o+1+(int)((h&0xFFFFFFFFULL)%(Q-1)))%Q;
          int dE=A[v*Q+n]-A[v*Q+o];
          if(dE<=0 || (double)(mix(h)>>11)*(1.0/9007199254740992.0)<exp(-dE/T)){
            q[v]=n; dEs+=dE;
            for(int k=ptr[v];k<ptr[v+1];k++){int u=nbr[k];
              #pragma omp atomic
              A[u*Q+o]--;
              #pragma omp atomic
              A[u*Q+n]++;}
          }
        }
      }
      #pragma omp single
      { if(trace&&sw%trace==0)printf("trace sweep=%d T=%.4f conflicts=%ld\n",sw,T,E+dEs); if(E+dEs==0){stop=1;used=sw+1;} }
      if(stop)break;
    }
    E+=dEs;
    double el=omp_get_wtime()-t;
    printf("seed=%d conflicts=%ld sweeps=%d time=%.3fs (%.1f us/sweep)\n",sd,E,used,el,1e6*el/used);
    if(E==0){ok++;swsum+=used;tsum+=el;}
  }
  printf("success %d/%d, mean sweeps to solve %.0f, mean time %.3fs\n",ok,seeds,ok?(double)swsum/ok:0,ok?tsum/ok:0);
}
