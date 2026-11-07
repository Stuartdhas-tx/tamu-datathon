// train.c - offline solver for heads-up short deck PLO5.
// Subcommands: preflop | sample | pool | train
// Build: clang -O3 -march=native -pthread train.c -o train -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define NP 630
#define NT 7140
static uint16_t *TAB;
static int PAIR[36][36], TRI[36][36][36];
static int C[40][7];

typedef struct { uint64_t s0, s1; } Rng;
static inline uint64_t rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static inline uint64_t rnd(Rng *r){ uint64_t a=r->s0,b=r->s1,res=a+b; b^=a; r->s0=rotl(a,24)^b^(b<<16); r->s1=rotl(b,37); return res; }
static inline uint32_t rndn(Rng *r, uint32_t n){ return (uint32_t)(((rnd(r)>>32)*(uint64_t)n)>>32); }
static inline float rndf(Rng *r){ return (float)(rnd(r)>>40)/(float)(1ull<<24); }
static void seed(Rng *r, uint64_t s){ s = s*0x9E3779B97F4A7C15ull+1; r->s0=s^0xBF58476D1CE4E5B9ull; r->s1=(s<<1)^0x94D049BB133111EBull; for(int i=0;i<16;i++) rnd(r); }

static void *map_file(const char *path, size_t *len, int write){
  int fd=open(path, write?O_RDWR:O_RDONLY); if(fd<0){perror(path);exit(1);}
  off_t n=lseek(fd,0,SEEK_END); void *p=mmap(0,n,PROT_READ|(write?PROT_WRITE:0),MAP_SHARED,fd,0);
  if(p==MAP_FAILED){perror("mmap");exit(1);} close(fd); if(len)*len=n; return p;
}
static void init_tables(const char *dir){
  char path[512]; snprintf(path,512,"%s/tab.bin",dir); TAB=map_file(path,0,0);
  int k=0; for(int a=0;a<36;a++)for(int b=a+1;b<36;b++){PAIR[a][b]=PAIR[b][a]=k++;}
  k=0; for(int a=0;a<36;a++)for(int b=a+1;b<36;b++)for(int c=b+1;c<36;c++){
    int v=k++; TRI[a][b][c]=TRI[a][c][b]=TRI[b][a][c]=TRI[b][c][a]=TRI[c][a][b]=TRI[c][b][a]=v; }
  for(int n=0;n<40;n++){C[n][0]=1;for(int r=1;r<7;r++)C[n][r]=n?C[n-1][r-1]+C[n-1][r]:0;}
}
static int b3[10][3]={{0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},{0,3,4},{1,2,3},{1,2,4},{1,3,4},{2,3,4}};
static int p2[10][2]={{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};

static inline void board_S(const int *b, uint16_t *S){
  memset(S,0,NP*2);
  for(int i=0;i<10;i++){ const uint16_t *row=TAB+(size_t)NP*TRI[b[b3[i][0]]][b[b3[i][1]]][b[b3[i][2]]];
    for(int j=0;j<NP;j++) if(row[j]>S[j]) S[j]=row[j]; }
}
static inline int hand_score(const int *h, const uint16_t *S){
  int best=0; for(int i=0;i<10;i++){int v=S[PAIR[h[p2[i][0]]][h[p2[i][1]]]]; if(v>best)best=v;} return best;
}
// rank of `mine` (5-card board) vs N random holdings from live[]
static inline float rank_vs_random(int mine, const uint16_t *S, int *live, int n, int N, Rng *r){
  int w=0;
  for(int s=0;s<N;s++){
    int h[5]; for(int i=0;i<5;i++){int j=i+rndn(r,n-i); int t=live[i];live[i]=live[j];live[j]=t;h[i]=live[i];}
    int v=hand_score(h,S); w+= (mine>v)?2:((mine==v)?1:0);
  }
  return w/(2.0f*N);
}
#define M_FLOP 60
#define K_FLOP 50
#define K_TURN 50
#define N_RIVER 1000
#define NU_T 0.9f

typedef struct { float fm,fn,tm,tn,rr; } Feat;
// board has 5 cards; player sees 3/4/5 of them
static void features(const int *h, const int *board, Rng *r, Feat *f){
  uint64_t dead=0; for(int i=0;i<5;i++)dead|=1ull<<h[i];
  uint16_t S[NP] __attribute__((aligned(64)));
  int live[36], n;
  // river
  uint64_t d=dead; for(int i=0;i<5;i++)d|=1ull<<board[i];
  n=0; for(int c=0;c<36;c++) if(!(d>>c&1)) live[n++]=c;
  board_S(board,S); f->rr=rank_vs_random(hand_score(h,S),S,live,n,N_RIVER,r);
  // turn: every river card
  d=dead; for(int i=0;i<4;i++)d|=1ull<<board[i];
  int cand[36],nc=0; for(int c=0;c<36;c++) if(!(d>>c&1)) cand[nc++]=c;
  float sum=0,nut=0;
  for(int i=0;i<nc;i++){
    int b[5]={board[0],board[1],board[2],board[3],cand[i]};
    n=0; for(int j=0;j<nc;j++) if(j!=i) live[n++]=cand[j];
    board_S(b,S); float x=rank_vs_random(hand_score(h,S),S,live,n,K_TURN,r); sum+=x; nut+=x>=NU_T;
  }
  f->tm=sum/nc; f->tn=nut/nc;
  // flop: sampled runouts
  d=dead; for(int i=0;i<3;i++)d|=1ull<<board[i];
  nc=0; for(int c=0;c<36;c++) if(!(d>>c&1)) cand[nc++]=c;
  sum=nut=0;
  for(int m=0;m<M_FLOP;m++){
    int i=rndn(r,nc), j=rndn(r,nc-1); if(j>=i)j++;
    int b[5]={board[0],board[1],board[2],cand[i],cand[j]};
    n=0; for(int k=0;k<nc;k++) if(k!=i&&k!=j) live[n++]=cand[k];
    board_S(b,S); float x=rank_vs_random(hand_score(h,S),S,live,n,K_FLOP,r); sum+=x; nut+=x>=NU_T;
  }
  f->fm=sum/M_FLOP; f->fn=nut/M_FLOP;
}
static void deal15(int *c, Rng *r){ int d[36]; for(int i=0;i<36;i++)d[i]=i;
  for(int i=0;i<15;i++){int j=i+rndn(r,36-i);int t=d[i];d[i]=d[j];d[j]=t;c[i]=d[i];} }
static int hand_index(const int *sorted){ int k=0; for(int i=0;i<5;i++)k+=C[sorted[i]][i+1]; return k; }
static void sort5(int *a){ for(int i=1;i<5;i++){int x=a[i],j=i-1;while(j>=0&&a[j]>x){a[j+1]=a[j];j--;}a[j+1]=x;} }

// ---------- preflop ----------
#define NH 376992
typedef struct { int th,nth; float *mu,*nu; int *reps; int nreps; } PfArg;
static int HANDS[NH][5];
static void *pf_worker(void *v){
  PfArg *a=v; Rng r; seed(&r,1000+a->th); uint16_t S[NP] __attribute__((aligned(64)));
  for(int ri=a->th;ri<a->nreps;ri+=a->nth){
    int id=a->reps[ri]; int *h=HANDS[id]; uint64_t dead=0; for(int i=0;i<5;i++)dead|=1ull<<h[i];
    int cand[36],nc=0; for(int c=0;c<36;c++) if(!(dead>>c&1)) cand[nc++]=c;
    float sum=0,nut=0; const int M=1500,K=40;
    for(int m=0;m<M;m++){
      int b[5]; for(int i=0;i<5;i++){int j=i+rndn(&r,nc-i);int t=cand[i];cand[i]=cand[j];cand[j]=t;b[i]=cand[i];}
      board_S(b,S); float x=rank_vs_random(hand_score(h,S),S,cand+5,nc-5,K,&r); sum+=x; nut+=x>=NU_T;
    }
    a->mu[id]=sum/M; a->nu[id]=nut/M;
  }
  return 0;
}
static int cmd_preflop(const char *dir,int nth){
  static int perms[24][4]; int np=0;
  for(int a=0;a<4;a++)for(int b=0;b<4;b++)if(b!=a)for(int c=0;c<4;c++)if(c!=a&&c!=b){int d=6-a-b-c;perms[np][0]=a;perms[np][1]=b;perms[np][2]=c;perms[np][3]=d;np++;}
  int n=0;
  for(int e=4;e<36;e++)for(int d=3;d<e;d++)for(int c=2;c<d;c++)for(int b=1;b<c;b++)for(int a=0;a<b;a++){
    int h[5]={a,b,c,d,e}; int id=hand_index(h); memcpy(HANDS[id],h,sizeof h); n++; }
  printf("hands %d\n",n);
  int *canon=malloc(NH*sizeof(int)), *reps=malloc(NH*sizeof(int)); int nreps=0;
  for(int i=0;i<NH;i++){ int best=NH;
    for(int p=0;p<24;p++){ int t[5]; for(int j=0;j<5;j++){int c=HANDS[i][j]; t[j]=(c&~3)|perms[p][c&3];} sort5(t); int id=hand_index(t); if(id<best)best=id; }
    canon[i]=best; if(best==i)reps[nreps++]=i; }
  printf("canonical %d\n",nreps);
  float *mu=calloc(NH,4),*nu=calloc(NH,4);
  pthread_t t[64]; PfArg a[64];
  for(int i=0;i<nth;i++){a[i]=(PfArg){i,nth,mu,nu,reps,nreps};pthread_create(&t[i],0,pf_worker,&a[i]);}
  for(int i=0;i<nth;i++)pthread_join(t[i],0);
  for(int i=0;i<NH;i++){mu[i]=mu[canon[i]];nu[i]=nu[canon[i]];}
  char path[512]; snprintf(path,512,"%s/pf_feat.bin",dir); FILE *f=fopen(path,"wb"); fwrite(mu,4,NH,f); fwrite(nu,4,NH,f); fclose(f);
  return 0;
}

// ---------- feature sample (for thresholds) ----------
typedef struct { int th; long n; Feat *out; } SArg;
static void *s_worker(void *v){ SArg *a=v; Rng r; seed(&r,77+a->th);
  for(long i=0;i<a->n;i++){ int c[15]; deal15(c,&r); features(c,c+10,&r,&a->out[i]); }
  return 0; }
static int cmd_sample(const char *dir,long n,int nth){
  Feat *out=malloc(n*sizeof(Feat)); pthread_t t[64]; SArg a[64]; long per=n/nth;
  for(int i=0;i<nth;i++){a[i]=(SArg){i,per,out+i*per};pthread_create(&t[i],0,s_worker,&a[i]);}
  for(int i=0;i<nth;i++)pthread_join(t[i],0);
  char path[512]; snprintf(path,512,"%s/feat_sample.bin",dir); FILE *f=fopen(path,"wb"); fwrite(out,sizeof(Feat),per*nth,f); fclose(f);
  return 0;
}

// ---------- pool ----------
// thresholds file format (float32): nM NN(per-level) then edges; written by buckets.py
typedef struct { int nm,nn; float me[64]; float ne[64][16]; } Grid;
static Grid GF,GT; static int NRE; static float RE[128];
static uint8_t *PFB;
static void read_grid(FILE *f, Grid *g){ float x; fread(&x,4,1,f); g->nm=(int)x; fread(&x,4,1,f); g->nn=(int)x;
  fread(g->me,4,g->nm-1,f); for(int i=0;i<g->nm;i++) fread(g->ne[i],4,g->nn-1,f); }
static void load_buckets(const char *dir){
  char path[512]; snprintf(path,512,"%s/thresholds.bin",dir); FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
  read_grid(f,&GF); read_grid(f,&GT); float x; fread(&x,4,1,f); NRE=(int)x; fread(RE,4,NRE-1,f); fclose(f);
  snprintf(path,512,"%s/pf_bucket.bin",dir); PFB=map_file(path,0,0);
}
static inline int lvl(float x,const float *e,int n){ int k=0; while(k<n-1 && x>=e[k])k++; return k; }
static inline int grid_b(const Grid *g,float m,float nu){ int a=lvl(m,g->me,g->nm); return a*g->nn+lvl(nu,g->ne[a],g->nn); }
typedef struct { uint8_t b[2][4]; uint8_t res; } __attribute__((packed)) DealRec; // res: 0 seat0 wins,1 seat1 wins,2 tie
typedef struct { int th; long n; DealRec *out; uint64_t sd; } PArg;
static void *p_worker(void *v){ PArg *a=v; Rng r; seed(&r,a->sd+a->th);
  uint16_t S[NP] __attribute__((aligned(64)));
  for(long i=0;i<a->n;i++){
    int c[15]; deal15(c,&r); DealRec *d=&a->out[i];
    for(int s=0;s<2;s++){ int h[5]; memcpy(h,c+5*s,sizeof h); sort5(h);
      Feat f; features(h,c+10,&r,&f);
      d->b[s][0]=PFB[hand_index(h)]; d->b[s][1]=grid_b(&GF,f.fm,f.fn); d->b[s][2]=grid_b(&GT,f.tm,f.tn); d->b[s][3]=lvl(f.rr,RE,NRE); }
    board_S(c+10,S); int x=hand_score(c,S),y=hand_score(c+5,S); d->res= x>y?0:(y>x?1:2);
    if(a->th==0 && i%100000==0){printf("pool %ld/%ld\n",i,a->n);fflush(stdout);}
  } return 0; }
static int cmd_pool(const char *dir,long n,int nth,const char *out,uint64_t sd){
  load_buckets(dir); DealRec *o=malloc(n*sizeof(DealRec)); pthread_t t[64]; PArg a[64]; long per=n/nth;
  for(int i=0;i<nth;i++){a[i]=(PArg){i,per,o+i*per,sd};pthread_create(&t[i],0,p_worker,&a[i]);}
  for(int i=0;i<nth;i++)pthread_join(t[i],0);
  FILE *f=fopen(out,"wb"); fwrite(o,sizeof(DealRec),per*nth,f); fclose(f); return 0;
}

// ---------- betting graph (mirrors game.py exactly) ----------
enum {FOLD=0,CALL=1,HALF=2,POT=3};
typedef struct { int32_t ch[4]; int32_t ctx; int16_t c0,c1; uint8_t street,pl,mask,term; } Node; // term:0 no,1 fold(pl folded),2 showdown
static Node *G; static int NG=0, CAPG=0;
typedef struct { uint8_t street,to,raises,acted,aggr,potb; int16_t ps; int16_t c[2]; int16_t b[2]; int16_t inc; uint32_t seq; } St;
#define HSZ (1<<26)
static int32_t *HT; static St *HK;
static inline uint64_t hst(const St *s){ uint64_t h=1469598103934665603ull; const uint8_t *p=(void*)s; for(size_t i=0;i<sizeof(St);i++){h^=p[i];h*=1099511628211ull;} return h; }
static int potb(int p){ static int e[]={4,6,8,12,18,26,36,50,72,100,159}; int k=0; while(k<10&&p>e[k])k++; return k; }
// context table
#define CHS (1<<22)
static uint64_t *CK; static int32_t *CV; static int NCTX=0; static int CTXSTREET[2000000];
static int ctx_id(int street,int pl,int pb,int aggr,uint32_t seq){
  uint64_t k=1+((uint64_t)seq<<20|(uint64_t)pb<<8|street<<4|pl<<3|aggr<<1);
  uint64_t h=(k*0x9E3779B97F4A7C15ull)>>42;
  while(CK[h]&&CK[h]!=k)h=(h+1)&(CHS-1);
  if(!CK[h]){CK[h]=k;CV[h]=NCTX;CTXSTREET[NCTX]=street;NCTX++;}
  return CV[h];
}
static int new_node(void){ if(NG==CAPG){CAPG=CAPG?CAPG*2:1<<16;G=realloc(G,CAPG*sizeof(Node));} memset(&G[NG],0,sizeof(Node)); return NG++; }
static int term_node(int c0,int c1,int type,int pl){ int n=new_node(); G[n].term=type; G[n].c0=c0; G[n].c1=c1; G[n].pl=pl; return n; }
static int build(St s);
static int after(St s,int acted_pl){ // non-fold action applied, s updated
  int both=(s.acted&3)==3 && s.b[0]==s.b[1];
  if(!both){ s.to=1-s.to; return build(s); }
  int st0=80-s.c[0], st1=80-s.c[1];
  if(s.street==3 || st0==0 || st1==0) return term_node(s.c[0],s.c[1],2,0);
  s.street++; s.b[0]=s.b[1]=0; s.acted=0; s.raises=0; s.inc=2; s.to=1; /* non-button (pl 1) first */
  s.ps=s.c[0]+s.c[1]; s.potb=potb(s.ps); s.seq=1;
  return build(s);
}
static int build(St s){
  uint8_t key[sizeof(St)]; (void)key;
  uint64_t h=hst(&s)&(HSZ-1);
  while(HT[h]>=0){ if(!memcmp(&HK[HT[h]],&s,sizeof(St))) return HK[HT[h]].ps<0?-1:((int32_t*)&HK[HSZ])[HT[h]]; h=(h+1)&(HSZ-1); }
  static int nk=0; int ki=nk++; if(ki>=HSZ-1){puts("hash full");exit(1);} HT[h]=ki; HK[ki]=s;
  int n=new_node(); ((int32_t*)&HK[HSZ])[ki]=n;
  int pl=s.to, opp=1-pl; int stack=80-s.c[pl], ostack=80-s.c[opp];
  int tc=s.b[opp]-s.b[pl]; if(tc<0)tc=0; int mask=2;
  if(tc>0) mask|=1;
  int canr = s.raises<4 && stack>tc && ostack>0;
  if(canr) mask|=12;
  int ctx=ctx_id(s.street,pl,s.ps,s.aggr,s.seq);
  int street=s.street;
  int ch[4]={-1,-1,-1,-1};
  if(tc>0){ ch[FOLD]=term_node(s.c[0],s.c[1],1,pl); }
  { St t=s; int amt=tc<stack?tc:stack; t.c[pl]+=amt; t.b[pl]+=amt; t.acted|=1<<pl; t.seq=s.seq*4+1; ch[CALL]=after(t,pl); }
  if(canr){
    int pot=s.c[0]+s.c[1], mn=s.b[opp]+(s.inc>2?s.inc:2), allin=s.b[pl]+stack, potto=s.b[opp]+pot+tc, half=s.b[opp]+(pot+tc)/2;
    for(int a=HALF;a<=POT;a++){
      int target=a==HALF?half:potto; int x=target>mn?target:mn; if(x>potto)x=potto; if(x>allin)x=allin;
      St t=s; int amt=x-s.b[pl]; t.c[pl]+=amt; t.b[pl]=x; t.inc=x-s.b[opp]; t.raises++; t.acted=(s.acted|(1<<pl))&~(1<<opp); t.seq=s.seq*4+a; t.aggr=pl+1;
      ch[a]=after(t,pl);
    }
  }
  Node *N=&G[n]; N->ctx=ctx; N->street=street; N->pl=pl; N->mask=mask; memcpy(N->ch,ch,sizeof ch); N->c0=s.c[0]; N->c1=s.c[1];
  return n;
}
static int build_graph(void){
  HT=malloc(HSZ*4); memset(HT,0xff,HSZ*4); HK=malloc(HSZ*sizeof(St)+HSZ*4);
  CK=calloc(CHS,8); CV=calloc(CHS,4);
  St s; memset(&s,0,sizeof s); s.street=0; s.to=0; s.c[0]=1; s.c[1]=2; s.b[0]=1; s.b[1]=2; s.inc=2; s.ps=3; s.potb=0; s.seq=1; s.aggr=0;
  int root=build(s); printf("nodes %d contexts %d\n",NG,NCTX);
  free(HT); free(HK); return root;
}

// ---------- MCCFR ----------
#define NB 128
static float *REG,*AVG; static DealRec *POOL; static long NPOOL; static int ROOT;
static float F_MULT=1,R_MULT=1; static int BR_MODE=0; static float *REGO=0; // restricted-response: opponent model tables
static inline size_t off(int ctx,int b){ return ((size_t)ctx*NB+b)*4; }
static inline void rm(const float *R,int mask,float *s){
  float sum=0; for(int a=0;a<4;a++) if(mask>>a&1){ float x=R[a]>0?R[a]:0; s[a]=x; sum+=x; } else s[a]=0;
  if(sum>0){ for(int a=0;a<4;a++) s[a]/=sum; }
  else { int n=__builtin_popcount(mask); for(int a=0;a<4;a++) s[a]=(mask>>a&1)?1.0f/n:0; }
}
// distort a model opponent's strategy: fold*F, raise*R (renormalised)
static inline void distort(float *s,int mask){
  if(mask&1) s[0]*=F_MULT; s[2]*=R_MULT; s[3]*=R_MULT;
  float t=s[0]+s[1]+s[2]+s[3]; if(t>0) for(int a=0;a<4;a++) s[a]/=t;
}
typedef struct { const DealRec *d; int seat[2]; Rng *r; int trav; int rr; } W; // seat[pl]=deal seat of pl (0=button)
#define RCLAMP (-3e6f)
static float walk(int n,const W *w){
  const Node *N=&G[n];
  if(N->term){
    float c[2]={N->c0,N->c1};
    if(N->term==1){ // pl folded: pl loses its contribution
      return N->pl==w->trav? -c[N->pl] : c[N->pl];
    }
    int res=w->d->res; if(res==2) return 0;
    int winpl = w->seat[0]==res?0:1;
    return winpl==w->trav? c[1-w->trav] : -c[w->trav];
  }
  int pl=N->pl; int b=w->d->b[w->seat[pl]][N->street];
  float s[4];
  if(pl==w->trav){
    float *R=REG+off(N->ctx,b);
    rm(R,N->mask,s);
    if(BR_MODE){ float *A=AVG+off(N->ctx,b); for(int a=0;a<4;a++) A[a]+=s[a]; }
    float u[4]={0,0,0,0},ev=0;
    for(int a=0;a<4;a++) if(N->mask>>a&1){ u[a]=walk(N->ch[a],w); ev+=s[a]*u[a]; }
    for(int a=0;a<4;a++) if(N->mask>>a&1){ float x=R[a]+(u[a]-ev); R[a]=x<RCLAMP?RCLAMP:x; }
    return ev;
  }
  float *R=(REGO&&w->rr?REGO:REG)+off(N->ctx,b);
  rm(R,N->mask,s);
  if(!(REGO&&w->rr)){ float *A=AVG+off(N->ctx,b); for(int a=0;a<4;a++) A[a]+=s[a]; }
  if(REGO&&w->rr) distort(s,N->mask);
  float x=rndf(w->r),acc=0; int pick=-1;
  for(int a=0;a<4;a++) if(N->mask>>a&1){ acc+=s[a]; pick=a; if(x<acc) break; }
  return walk(N->ch[pick],w);
}
// In restricted mode there are two learners. Learner L: REG/AVG. Model M: REGO.
// rr==1: traverse L; opponent nodes use distorted M.  rr==2 below: traverse M vs L.
static float walk_model(int n,const W *w){
  const Node *N=&G[n];
  if(N->term){ float c[2]={N->c0,N->c1};
    if(N->term==1) return N->pl==w->trav? -c[N->pl] : c[N->pl];
    int res=w->d->res; if(res==2) return 0; int winpl=w->seat[0]==res?0:1;
    return winpl==w->trav? c[1-w->trav] : -c[w->trav]; }
  int pl=N->pl; int b=w->d->b[w->seat[pl]][N->street]; float s[4];
  if(pl==w->trav){ float *R=REGO+off(N->ctx,b); rm(R,N->mask,s); float u[4]={0,0,0,0},ev=0;
    for(int a=0;a<4;a++) if(N->mask>>a&1){u[a]=walk_model(N->ch[a],w);ev+=s[a]*u[a];}
    for(int a=0;a<4;a++) if(N->mask>>a&1){float x=R[a]+(u[a]-ev);R[a]=x<RCLAMP?RCLAMP:x;}
    return ev; }
  float *R=REG+off(N->ctx,b); rm(R,N->mask,s);
  { float *A=AVG+off(N->ctx,b); for(int a=0;a<4;a++) A[a]+=s[a]; }
  float x=rndf(w->r),acc=0; int pick=-1;
  for(int a=0;a<4;a++) if(N->mask>>a&1){acc+=s[a];pick=a;if(x<acc)break;}
  return walk_model(N->ch[pick],w);
}
typedef struct { int th; long iters; uint64_t sd; double sum; } TArg;
static void *t_worker(void *v){ TArg *a=v; Rng r; seed(&r,a->sd); double sum=0;
  for(long i=0;i<a->iters;i++){
    const DealRec *d=&POOL[rnd(&r)%NPOOL]; int sw=rnd(&r)&1; W w; w.d=d; w.seat[0]=sw; w.seat[1]=1-sw; w.r=&r; 
    for(int t=0;t<2;t++){ w.trav=t; w.rr=REGO?1:0; sum+=walk(ROOT,&w); }
    if(REGO&&!BR_MODE){
      float m=F_MULT; (void)m;
      for(int t=0;t<2;t++){ w.trav=t; w.rr=0; walk_model(ROOT,&w);
        // keep the learner's average tied to its own play against the model
      }
    }
  }
  a->sum=sum; return 0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void dump(const char *out){
  size_t nf=(size_t)NCTX*NB*4; FILE *f=fopen(out,"wb"); fwrite(AVG,4,nf,f); fclose(f);
  char p[512]; snprintf(p,512,"%s.reg",out); f=fopen(p,"wb"); fwrite(REG,4,nf,f); fclose(f);
}
static void dump_graph(const char *dir){
  char p[512]; snprintf(p,512,"%s/ctx.txt",dir); FILE *f=fopen(p,"w");
  for(int h=0;h<CHS;h++) if(CK[h]) fprintf(f,"%llu %d\n",(unsigned long long)CK[h],CV[h]); fclose(f);
}
static int cmd_train(int argc,char **argv){
  // train <dir> <pool> <out> <iters_per_round> <rounds> <threads> [fmult rmult warmstart]
  const char *dir=argv[0],*pool=argv[1],*out=argv[2]; long per=atol(argv[3]); int rounds=atoi(argv[4]),nth=atoi(argv[5]);
  size_t len; POOL=map_file(pool,&len,0); NPOOL=len/sizeof(DealRec);
  ROOT=build_graph(); dump_graph(dir);
  size_t nf=(size_t)NCTX*NB*4; printf("floats %zu (%.0f MB) pool %ld\n",nf,nf*8/1e6,NPOOL);
  REG=calloc(nf,4); AVG=calloc(nf,4);
  if(argc>6 && !getenv("BR_VS")){ F_MULT=atof(argv[6]); R_MULT=atof(argv[7]); REGO=calloc(nf,4);
    if(argc>8){ char p[512]; snprintf(p,512,"%s.reg",argv[8]); FILE *f=fopen(p,"rb"); if(!f||fread(REG,4,nf,f)!=nf){perror(p);exit(1);} fclose(f);
      memcpy(REGO,REG,nf*4); f=fopen(argv[8],"rb"); fread(AVG,4,nf,f); fclose(f); for(size_t i=0;i<nf;i++){REG[i]*=0.01f;REGO[i]*=0.01f;AVG[i]=0;} } }
  if(getenv("BR_VS")){ // freeze the opponent at a saved average strategy
    BR_MODE=1; REGO=calloc(nf,4); FILE *f=fopen(getenv("BR_VS"),"rb"); if(!f||fread(REGO,4,nf,f)!=nf){perror("BR_VS");exit(1);} fclose(f);
    F_MULT=argc>6?atof(argv[6]):1; R_MULT=argc>7?atof(argv[7]):1;
    size_t i; for(i=0;i<nf;i+=4){ float t=REGO[i]+REGO[i+1]+REGO[i+2]+REGO[i+3]; if(t>0) for(int a=0;a<4;a++) REGO[i+a]/=t; }
  }
  double t0=now(); int disc_rounds=rounds*2/5;
  for(int rd=0;rd<rounds;rd++){
    pthread_t t[64]; TArg a[64];
    for(int i=0;i<nth;i++){a[i]=(TArg){i,per/nth,(uint64_t)rd*1000+i+1,0};pthread_create(&t[i],0,t_worker,&a[i]);}
    double s=0; for(int i=0;i<nth;i++){pthread_join(t[i],0);s+=a[i].sum;}
    if(rd<disc_rounds){ float d=(rd+1.0f)/(rd+2.0f); for(size_t i=0;i<nf;i++){REG[i]*=d;AVG[i]*=d;} if(REGO&&!BR_MODE) for(size_t i=0;i<nf;i++)REGO[i]*=d; }
    printf("round %d/%d  iters %.1fM  %.0fs  avgv %.4f\n",rd+1,rounds,(rd+1)*per/1e6,now()-t0,s/(per*2));fflush(stdout);
    if((rd+1)%10==0||rd==rounds-1) dump(out);
  }
  return 0;
}
int main(int argc,char **argv){
  if(argc<3){puts("usage");return 1;}
  init_tables(argv[2]);
  if(!strcmp(argv[1],"preflop")) return cmd_preflop(argv[2],atoi(argv[3]));
  if(!strcmp(argv[1],"sample")) return cmd_sample(argv[2],atol(argv[3]),atoi(argv[4]));
  if(!strcmp(argv[1],"pool")) return cmd_pool(argv[2],atol(argv[3]),atoi(argv[4]),argv[5],atoll(argv[6]));
  if(!strcmp(argv[1],"train")) return cmd_train(argc-2,argv+2);
  if(!strcmp(argv[1],"graph")){ build_graph(); dump_graph(argv[2]); return 0; }
  return 1;
}
