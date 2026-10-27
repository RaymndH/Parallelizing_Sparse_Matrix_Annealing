// Belief propagation for q=5 colouring (antiferromagnetic Potts at inverse temperature BETA), OpenMP.
// usage: OMP_NUM_THREADS=16 ./bp graph iters [plant.txt]
//   message from u to i:  psi(c) ~ prod_{k in nbr(u)\i} (1 - x*psi_{k->u}(c)),  x = 1-exp(-BETA)
//   synchronous updates with damping; marginals b_i(c) ~ prod_k (1 - x*psi_{k->i}(c)).
//   Estimate = argmax b_i. Reports conflicts, overlap with the planted colouring (best colour permutation,
//   if plant.txt is given) and, optionally, conflicts after a zero-temperature greedy clean-up.
//   env BETA (default 20; large = hard constraints), DAMP (default 0.5), CLEAN (greedy sweeps, default 50),
//   TRACE=k prints every k iterations. Max degree must be < 64.
// build: gcc -O3 -march=native -fopenmp -o bp bp.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#define Q 5
static inline uint64_t mix(uint64_t z){ z+=0x9E3779B97F4A7C15ULL; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"r"); int N; char line[256]; if(!fgets(line,sizeof line,f))return 1;
  if(sscanf(line,"N = %d",&N)!=1 && sscanf(line,"%d",&N)!=1){fprintf(stderr,"bad header\n");return 1;}
  int M=0,cap=1<<21,*ea=malloc(cap*4),*eb=malloc(cap*4),a,b;
  while(fscanf(f,"%d %d",&a,&b)==2){ if(M==cap){cap*=2;ea=realloc(ea,cap*4);eb=realloc(eb,cap*4);} ea[M]=a;eb[M]=b;M++;}
  int*deg=calloc(N+1,4),*ptr=calloc(N+1,4),*nbr=malloc(2*(size_t)M*4),*fill=calloc(N,4),*rev=malloc(2*(size_t)M*4);
  for(int i=0;i<M;i++){deg[ea[i]]++;deg[eb[i]]++;}
  for(int i=0;i<N;i++)ptr[i+1]=ptr[i]+deg[i];
  for(int i=0;i<M;i++){ int p=ptr[ea[i]]+fill[ea[i]]++, r=ptr[eb[i]]+fill[eb[i]]++; nbr[p]=eb[i]; nbr[r]=ea[i]; rev[p]=r; rev[r]=p; }
  for(int i=0;i<N;i++)if(deg[i]>=64){fprintf(stderr,"degree %d too large\n",deg[i]);return 1;}
  int iters=atoi(argv[2]);
  double beta=getenv("BETA")?atof(getenv("BETA")):20, damp=getenv("DAMP")?atof(getenv("DAMP")):0.5;
  int clean=getenv("CLEAN")?atoi(getenv("CLEAN")):50, trace=getenv("TRACE")?atoi(getenv("TRACE")):0;
  double x=1-exp(-beta); size_t E2=2*(size_t)M;
  int*plant=NULL;
  if(argc>3){ FILE*pf=fopen(argv[3],"r"); plant=malloc(N*4); for(int i=0;i<N;i++) if(fscanf(pf,"%d",&plant[i])!=1){fprintf(stderr,"short plant file\n");return 1;} fclose(pf); }
  setvbuf(stdout,NULL,_IOLBF,0);
  // m[e] (e in node i's CSR range) = message from nbr[e] to i
  float*m=malloc(E2*Q*4),*mn=malloc(E2*Q*4); float*bel=malloc((size_t)N*Q*4); int*col=malloc(N*4);
  #pragma omp parallel for
  for(size_t e=0;e<E2;e++){ double s=0; float t[Q]; for(int c=0;c<Q;c++){ t[c]=1.0f+0.1f*(float)((mix(e*Q+c+12345)>>40)*(1.0/16777216.0)); s+=t[c]; } for(int c=0;c<Q;c++)m[e*Q+c]=t[c]/s; }
  double t0=omp_get_wtime();
  int it,final_it=iters;
  for(it=0;it<iters;it++){
    double diff=0;
    #pragma omp parallel for schedule(static) reduction(+:diff)
    for(int i=0;i<N;i++){
      int d=deg[i]; float fac[64][Q]; float pre[Q],suf[64+1][Q];
      for(int k=0;k<d;k++)for(int c=0;c<Q;c++)fac[k][c]=1.0f-(float)x*m[(size_t)(ptr[i]+k)*Q+c];
      for(int c=0;c<Q;c++){pre[c]=1;suf[d][c]=1;}
      for(int k=d-1;k>=0;k--)for(int c=0;c<Q;c++)suf[k][c]=suf[k+1][c]*fac[k][c];
      for(int k=0;k<d;k++){
        float o[Q],s=0; for(int c=0;c<Q;c++){ o[c]=pre[c]*suf[k+1][c]; s+=o[c]; }
        if(s<=0){ for(int c=0;c<Q;c++)o[c]=1.0f/Q; s=1; }
        size_t r=(size_t)rev[ptr[i]+k]*Q;     // slot of this message at nbr[k]
        for(int c=0;c<Q;c++){ float nv=o[c]/s; float old=m[r+c]; float v=(float)(damp*old+(1-damp)*nv); mn[r+c]=v; diff+=fabsf(v-old); }
        for(int c=0;c<Q;c++)pre[c]*=fac[k][c];
      }
    }
    float*tmp=m;m=mn;mn=tmp;
    if(trace&&it%trace==0){ printf("iter %d mean|dpsi|=%.3e  t=%.2fs\n",it,diff/((double)E2*Q),omp_get_wtime()-t0); }
    if(diff/((double)E2*Q)<1e-9){final_it=it+1;break;}
  }
  double el=omp_get_wtime()-t0;
  #pragma omp parallel for schedule(static)
  for(int i=0;i<N;i++){
    double bb[Q],bs=0; for(int c=0;c<Q;c++)bb[c]=1;
    for(int k=ptr[i];k<ptr[i+1];k++)for(int c=0;c<Q;c++)bb[c]*=1.0-x*m[(size_t)k*Q+c];
    int best=0; for(int c=0;c<Q;c++){ bs+=bb[c]; if(bb[c]>bb[best])best=c; } col[i]=best;
    for(int c=0;c<Q;c++)bel[(size_t)i*Q+c]=(float)(bb[c]/(bs>0?bs:1));
  }
  long E=0; for(int i=0;i<M;i++)E+=col[ea[i]]==col[eb[i]];
  double maxb=0; for(int i=0;i<N;i++){ float mb=0; for(int c=0;c<Q;c++)if(bel[(size_t)i*Q+c]>mb)mb=bel[(size_t)i*Q+c]; maxb+=mb; }
  printf("BP: %d iters in %.2fs (%.1f ms/iter)  conflicts=%ld (%.3f%% of edges)  mean max-marginal=%.3f\n",final_it,el,1e3*el/final_it,E,100.0*E/M,maxb/N);
  if(plant){
    long cm[Q][Q]; memset(cm,0,sizeof cm); for(int i=0;i<N;i++)cm[plant[i]][col[i]]++;
    int perm[Q]={0,1,2,3,4},bestm=0; // brute-force the 120 permutations
    int idx[Q]; for(idx[0]=0;idx[0]<Q;idx[0]++)for(idx[1]=0;idx[1]<Q;idx[1]++)for(idx[2]=0;idx[2]<Q;idx[2]++)for(idx[3]=0;idx[3]<Q;idx[3]++)for(idx[4]=0;idx[4]<Q;idx[4]++){
      int ok=1; for(int a=0;a<Q&&ok;a++)for(int b2=a+1;b2<Q;b2++)if(idx[a]==idx[b2]){ok=0;break;}
      if(!ok)continue; long mt=0; for(int a=0;a<Q;a++)mt+=cm[a][idx[a]]; if(mt>bestm){bestm=(int)mt;(void)perm;} }
    printf("overlap with planted = %.4f (chance 0.2 -> 0, 1.0 solved: normalised %.4f)\n",(double)bestm/N,((double)bestm/N-1.0/Q)/(1-1.0/Q));
  }
  if(clean>0){   // zero-temperature greedy: move each node to its least-conflicting colour (sequential); conflicting nodes may also take an equal-cost colour
    int*q=col; int moved=1,sw;
    for(sw=0;sw<clean&&E>0;sw++){ moved=0;
      for(int v=0;v<N;v++){ int cnt[Q]={0}; for(int k=ptr[v];k<ptr[v+1];k++)cnt[q[nbr[k]]]++;
        int bc=q[v]; for(int c=0;c<Q;c++)if(cnt[c]<cnt[bc])bc=c;
        if(bc==q[v]&&cnt[q[v]]>0){ // plateau move: a conflicting node may switch to an equally good colour
          int nt=0,cand[Q]; for(int c=0;c<Q;c++)if(c!=q[v]&&cnt[c]==cnt[q[v]])cand[nt++]=c;
          if(nt)bc=cand[mix(v+(uint64_t)sw*N)%nt]; }
        if(bc!=q[v]){E-=cnt[q[v]]-cnt[bc];q[v]=bc;moved++;} } }
    printf("after %d greedy sweeps: conflicts=%ld  %s\n",sw,E,E==0?"SOLVED":"");
  }
  return 0;
}
