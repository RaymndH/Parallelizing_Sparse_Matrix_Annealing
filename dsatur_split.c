// Write independent-set splits made with DSATUR (random tie-breaking, so every split differs).
// usage: ./dsatur_split graph n_splits out.txt   (one line of N class labels per split, as make_splits.py)
// build: gcc -O3 -march=native -o dsatur_split dsatur_split.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s=88172645463325252ULL; static inline uint64_t rnd(){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
typedef struct{int sat,deg;uint32_t r;int v;} H;
static H*h; static long hn;
static int better(H a,H b){ if(a.sat!=b.sat)return a.sat>b.sat; if(a.deg!=b.deg)return a.deg>b.deg; return a.r>b.r; }
static void push(H x){ long i=hn++; h[i]=x; while(i>0){long p=(i-1)/2; if(!better(h[i],h[p]))break; H t=h[i];h[i]=h[p];h[p]=t;i=p;} }
static H pop(){ H top=h[0]; h[0]=h[--hn]; long i=0; for(;;){long l=2*i+1,r=l+1,b=i; if(l<hn&&better(h[l],h[b]))b=l; if(r<hn&&better(h[r],h[b]))b=r; if(b==i)break; H t=h[i];h[i]=h[b];h[b]=t;i=b;} return top; }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"r"); int ns=atoi(argv[2]),N; char line[256]; if(!fgets(line,sizeof line,f))return 1;
  if(sscanf(line,"N = %d",&N)!=1 && sscanf(line,"%d",&N)!=1)return 1;
  long M=0,cap=1<<21; int*ea=malloc(cap*4),*eb=malloc(cap*4),a,b;
  while(fscanf(f,"%d %d",&a,&b)==2){ if(M==cap){cap*=2;ea=realloc(ea,cap*4);eb=realloc(eb,cap*4);} ea[M]=a;eb[M]=b;M++; }
  int*deg=calloc(N,4),*ptr=calloc(N+1,4),*nbr=malloc(2*M*4),*fill=calloc(N,4);
  for(long i=0;i<M;i++){deg[ea[i]]++;deg[eb[i]]++;}
  for(int i=0;i<N;i++)ptr[i+1]=ptr[i]+deg[i];
  for(long i=0;i<M;i++){nbr[ptr[ea[i]]+fill[ea[i]]++]=eb[i];nbr[ptr[eb[i]]+fill[eb[i]]++]=ea[i];}
  int*col=malloc(N*4); uint64_t*mask=malloc(N*8); h=malloc(((size_t)N+2*M+16)*sizeof(H));
  FILE*o=fopen(argv[3],"w"); int ksum=0;
  for(int t=0;t<ns;t++){
    int K=0; hn=0; s^=0x9E3779B97F4A7C15ULL*(t+1); rnd();
    for(int i=0;i<N;i++){col[i]=-1;mask[i]=0; H x={0,deg[i],(uint32_t)rnd(),i}; push(x);}
    int done=0;
    while(done<N){ H x=pop(); int v=x.v; if(col[v]>=0||x.sat!=__builtin_popcountll(mask[v]))continue;
      int c=0; while(mask[v]>>c&1)c++; col[v]=c; done++; if(c+1>K)K=c+1;
      for(int e=ptr[v];e<ptr[v+1];e++){int u=nbr[e]; if(col[u]>=0)continue; uint64_t m=mask[u]|(1ULL<<c);
        if(m!=mask[u]){mask[u]=m; H y={__builtin_popcountll(m),deg[u],(uint32_t)rnd(),u}; push(y);}}}
    for(long e=0;e<M;e++) if(col[ea[e]]==col[eb[e]]){fprintf(stderr,"INVALID split\n");return 1;}
    for(int i=0;i<N;i++)fprintf(o,i?" %d":"%d",col[i]); fprintf(o,"\n"); ksum+=K;
  }
  fclose(o); printf("wrote %d DSATUR splits for N=%d, mean %.2f sets\n",ns,N,(double)ksum/ns);
}
