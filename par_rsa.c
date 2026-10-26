// Replicated simulated annealing (Angelini and Ricci-Tersenghi, arXiv 2206.04760) for q=5 colouring, OpenMP.
// usage: OMP_NUM_THREADS=16 ./par_rsa graph T0 T1 sweeps runs y gamma [splits.txt]
//   `runs` independent runs, each made of y coupled replicas with energy
//     H = sum_a (conflicting edges of replica a) - gamma/(2y) * sum_{a!=b} #{nodes with equal colour in a and b}.
//   One sweep visits the replicas of a run one after another (replica k: every independent set in turn, nodes of a set
//   in parallel), so it equals a sequential sweep. Solved replicas (0 conflicts) are frozen; a run is solved when all
//   its replicas are. With y=1 the result is identical to par.c. Random numbers hash (seed, sweep, replica, node).
//   splits.txt as in par.c (e.g. from dsatur_split); env LIN=1 linear schedule, TRACE=k prints min/mean conflicts.
// build: gcc -O3 -march=native -fopenmp -o par_rsa par_rsa.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
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
  double T0=atof(argv[2]),T1=atof(argv[3]); int sweeps=atoi(argv[4]),seeds=atoi(argv[5]); int Y=atoi(argv[6]); double gamma=atof(argv[7]); double coup=gamma/Y;
  int lin=getenv("LIN")!=NULL, trace=getenv("TRACE")?atoi(getenv("TRACE")):0;
  setvbuf(stdout,NULL,_IOLBF,0);
  // splits: order[sp] lists nodes class by class, start[sp][c] gives class offsets
  int*order=malloc((size_t)NS*N*4),*start[NS],K[NS];
  s=88172645463325252ULL; FILE*sf=argc>8?fopen(argv[8],"r"):NULL;
  { int*perm=malloc(N*4),*col=malloc(N*4),*cnt=malloc((size_t)(N+1)*4);
    for(int sp=0;sp<NS;sp++){
      int k=0;
      if(sf){ for(int i=0;i<N;i++){ if(fscanf(sf,"%d",&col[i])!=1){fprintf(stderr,"short splits file\n");return 1;} if(col[i]+1>k)k=col[i]+1; } }
      else {
      for(int i=0;i<N;i++)perm[i]=i;
      for(int i=N-1;i>0;i--){int j=rnd()%(i+1),x=perm[i];perm[i]=perm[j];perm[j]=x;}
      for(int i=0;i<N;i++)col[i]=-1;
      for(int ii=0;ii<N;ii++){int v=perm[ii]; unsigned long long used=0;
        for(int e=ptr[v];e<ptr[v+1];e++)if(col[nbr[e]]>=0&&col[nbr[e]]<64)used|=1ULL<<col[nbr[e]];
        int c=0; while(used>>c&1)c++; col[v]=c; if(c+1>k)k=c+1;}
      }
      K[sp]=k; start[sp]=calloc(k+1,4);
      for(int i=0;i<N;i++)start[sp][col[i]+1]++;
      for(int c=0;c<k;c++)start[sp][c+1]+=start[sp][c];
      for(int c=0;c<k;c++)cnt[c]=start[sp][c];
      for(int i=0;i<N;i++)order[(size_t)sp*N+cnt[col[i]]++]=i;
    }
    printf("threads=%d  %s  sets/split=",omp_get_max_threads(),sf?"splits from file":"greedy splits"); for(int sp=0;sp<NS;sp++)printf("%d ",K[sp]); printf("\n");
  }
  int*q=malloc((size_t)Y*N*4),*A=calloc((size_t)Y*N*Q,4); long*Ek=malloc(Y*sizeof(long)); int*frozen=calloc(Y,4);
  int ok=0; double tsum=0; long swsum=0;
  printf("runs=%d y=%d gamma=%g\n",seeds,Y,gamma);
  for(int sd=0;sd<seeds;sd++){
    uint64_t seed=mix(0x1234567ULL+sd); s=seed|1;
    for(size_t i=0;i<(size_t)Y*N;i++)q[i]=rnd()%Q;
    memset(A,0,(size_t)Y*N*Q*4);
    for(int k=0;k<Y;k++){ int*qk=q+(size_t)k*N,*Ak=A+(size_t)k*N*Q;
      for(int i=0;i<N;i++)for(int e=ptr[i];e<ptr[i+1];e++)Ak[i*Q+qk[nbr[e]]]++;
      long E=0; for(long e=0;e<M;e++)E+=qk[ea[e]]==qk[eb[e]]; Ek[k]=E; frozen[k]=(E==0); }
    long dEs=0; int stop=0,used=sweeps,sp=0,lastsw=0; double T=T0;
    double t=omp_get_wtime(),tlast=t;
    #pragma omp parallel
    for(int sw=0;sw<sweeps;sw++){
      #pragma omp single
      { T=lin?T0-(T0-T1)*(double)sw/(sweeps-1):T0*pow(T1/T0,(double)sw/(sweeps-1)); sp=mix(seed^(uint64_t)sw)%NS; }
      for(int k=0;k<Y;k++){
        if(frozen[k])continue;                       // changed only inside single blocks, so all threads agree
        int*qk=q+(size_t)k*N,*Ak=A+(size_t)k*N*Q;
        for(int c=0;c<K[sp];c++){
          const int*ord=order+(size_t)sp*N+start[sp][c]; int n_c=start[sp][c+1]-start[sp][c];
          #pragma omp for schedule(static) reduction(+:dEs)
          for(int ii=0;ii<n_c;ii++){
            int v=ord[ii],o=qk[v]; uint64_t h=mix(seed^((((uint64_t)sw*Y+k)*(uint64_t)N)+v)*0x9E3779B97F4A7C15ULL);
            int n=(o+1+(int)((h&0xFFFFFFFFULL)%(Q-1)))%Q;
            int dEd=Ak[v*Q+n]-Ak[v*Q+o];             // change of this replica's conflicts
            int cn=0,co=0; for(int m=0;m<Y;m++)if(m!=k){int cm=q[(size_t)m*N+v]; cn+=cm==n; co+=cm==o;}
            double dE=dEd-coup*(cn-co);              // plus replica coupling (other replicas are not being updated)
            if(dE<=0 || (double)(mix(h)>>11)*(1.0/9007199254740992.0)<exp(-dE/T)){
              qk[v]=n; dEs+=dEd;
              for(int e=ptr[v];e<ptr[v+1];e++){int u=nbr[e];
                #pragma omp atomic
                Ak[u*Q+o]--;
                #pragma omp atomic
                Ak[u*Q+n]++;}
            }
          }
        }
        #pragma omp single
        { Ek[k]+=dEs; dEs=0; if(Ek[k]==0)frozen[k]=1; }
      }
      #pragma omp single
      { if(trace&&sw%trace==0){ double now=omp_get_wtime(); int dsw=sw-lastsw>0?sw-lastsw:1; long mn=Ek[0],sm=0; for(int k=0;k<Y;k++){if(Ek[k]<mn)mn=Ek[k]; sm+=Ek[k];}
          printf("trace sweep=%d T=%.4f conflicts_min=%ld conflicts_mean=%.1f us/sweep=%.1f\n",sw,T,mn,(double)sm/Y,1e6*(now-tlast)/dsw); lastsw=sw; tlast=now; }
        int all=1; for(int k=0;k<Y;k++)if(Ek[k]!=0)all=0; if(all){stop=1;used=sw+1;} }
      if(stop)break;
    }
    double el=omp_get_wtime()-t; int all=1;
    for(int k=0;k<Y;k++){ int*qk=q+(size_t)k*N; long E=0; for(long e=0;e<M;e++)E+=qk[ea[e]]==qk[eb[e]];
      if(E!=Ek[k]){fprintf(stderr,"INCONSISTENT replica %d: tracked %ld, recomputed %ld\n",k,Ek[k],E);return 1;} if(E)all=0; }
    printf("seed=%d conflicts=[",sd); for(int k=0;k<Y;k++)printf(k?",%ld":"%ld",Ek[k]); printf("] sweeps=%d time=%.3fs (%.1f us/sweep)\n",used,el,1e6*el/used);
    if(all){ok++;swsum+=used;tsum+=el;}
  }
  printf("success %d/%d, mean sweeps to solve %.0f, mean time %.3fs\n",ok,seeds,ok?(double)swsum/ok:0,ok?tsum/ok:0);
}
