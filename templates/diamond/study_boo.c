#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "vector.h"
#include "global_definitions.h"
#include "io.h"
#include "interaction_map.h"
#include "events.h"
#include "cluster.h"
#include "cell_list.h"
#include "dirutils.h"
#include "smart_allocator.h"
#include "ordinator.h"
#include "bilista.h"
#include "random.h"
#include "log.h"
#include "order_parameters.h"


#define SQR(x) ((x)*(x))
#define SCALAR(v,u) (((v)->x)*((u)->x)+((v)->y)*((u)->y)+((v)->z)*((u)->z))


#define LIQUID_CODE 0
#define DC_CODE 1
#define IH_CODE 2
#define CLATRATO_CODE 3
#define T12_CODE 4
#define SC16_CODE 5
#define II_CODE 6


typedef struct _clusterdistribution {
	int id;
	int root;
	vector cm;
	int *particles;
	int num_particles;
} clusterdistribution;

typedef struct _compactcluster {
	int *label;
	int *whoaddedme;
	int num;
} compactcluster;

void readPositionsOXDNA(char *_input_name,vector *pos,steps *time,int *numparticles,double NOBox[],double INOBox[]);
int getNumberParticlesOXDNA(char *_input_name);

int estraiTempo(char *nomefile)
{
        char copy[100];
        strcpy(copy,nomefile);
        
        /* questa funzione cambia a seconda di come sono salvati i file */
        
        /* formato attuale pos_$(time) */
        strtok(copy,"_");
        char *token=strtok(NULL,"_");
        
        // check
        if (strtok(NULL,"_")!=NULL)
        {
                printf("Error: are you using the right way to extract times?");
                exit(1);
        }
        
        return atoi(token);
}


// Frenkel-Smit algorithm
void randomVersor(vector *rv,gsl_rng *random)
{
	double ransq=2.;
	double ran1,ran2;
	double ranh;
	
	while (ransq>=1.)
	{
		ran1=1.-2.*gsl_rng_uniform(random);
		ran2=1.-2.*gsl_rng_uniform(random);
		ransq=SQR(ran1)+SQR(ran2);
	}
	
	ranh=2.*sqrt(1.-ransq);
	
	rv->x=ran1*ranh;
	rv->y=ran2*ranh;
	rv->z=1.-2.*ransq;
	
}


vector getPerpendicularVersor(vector *v1,gsl_rng *Gsl_random)
{
	vector v2;
	
	randomVersor(&v2,Gsl_random);
	
	double v1_norm2=SQR(v1->x)+SQR(v1->y)+SQR(v1->z);
	double v2_v1=v2.x*v1->x+v2.y*v1->y+v2.z*v1->z;
	double buffer=v2_v1/v1_norm2;
	
	// u2=v2-((v2*u1)/(u1.norma**2))*u1
	v2.x=v2.x-(buffer)*v1->x;
	v2.y=v2.y-(buffer)*v1->y;
	v2.z=v2.z-(buffer)*v1->z;
	
	double inorma=1./sqrt(SQR(v2.x)+SQR(v2.y)+SQR(v2.z));
	
	v2.x*=inorma;
	v2.y*=inorma;
	v2.z*=inorma;
	
	return v2;
}

vector vectorVectorProduct (const vector *v1,const vector *v2)
{
	vector result;
	
	result.x=(v1->y*v2->z)-(v1->z*v2->y);
	result.y=(v1->z*v2->x)-(v1->x*v2->z);
	result.z=(v1->x*v2->y)-(v1->y*v2->x);
	
	return result;
	
}


static int compareInt(const void *node1,const void *node2)
{
	if ( (*(int*)node1) > (*(int*)node2) ) return 1;
	if ( (*(int*)node1) < (*(int*)node2) ) return -1;
	else
		return 0;
}

static void insertionSort(int *v,int *length,int num)
{
	int l=*length;
	v[l]=num;
	while ((l>0) && (v[l]<v[l-1]))
	{
		int buffer;
		buffer=v[l];
		v[l]=v[l-1];
		v[l-1]=buffer;
		l--;
	}
	(*length)++;
}


static void pbcNearestImage(vector *image,const vector *origin,double Box[])
{
	vector olddist;
	
	olddist.x=image->x-origin->x;
	olddist.y=image->y-origin->y;
	olddist.z=image->z-origin->z;
	
	image->x-=rint(olddist.x);
	image->y-=rint(olddist.y);
	image->z-=rint(olddist.z);
}



int main(int argc,char *argv[])
{
	
	if (argc!=6)
	{
		printf("%s [op range=2.5] [maxneighbours=16] [threshold=0.75] [connections=12] [input file]\n",argv[0]);
		exit(1);
	}
	
	gsl_rng *Gsl_random=randomConstructorInteractive(0);
	
	double limit_Q4_clathrate=0.05;
	double limit_Q4_supercooled=0.11;
	double limit_Q4_t12=0.145;
	
	
	double range=atof(argv[1]);
	int maxneighbours=atoi(argv[2]);
	double threshold=atof(argv[3]);
	int numconnections=atoi(argv[4]);
	char input[100];
	strcpy(input,argv[5]);
	
	
	int ncolloids=getNumberParticlesOXDNA(input);
	
	vector *pos=calloc(ncolloids,sizeof(vector));
	
	steps time;
	double box[6],ibox[6];
	
	readPositionsOXDNA(input,pos,&time,&ncolloids,box,ibox);
	
	listcell *cells=getList(box,range,ncolloids);
	fullUpdateList(cells,pos,ncolloids,box,range);
	
	orderparam *op=allocateOP(ncolloids,2,1000);
	//request_OpL('q',3,ncolloids,op);
	request_OpL('Q',4,ncolloids,op);
	request_OpL('W',4,ncolloids,op);
	request_OpL('Q',12,ncolloids,op);
	
	
	vector X_versor,Y_versor,Z_versor;
	randomVersor(&X_versor,Gsl_random);
	Y_versor=getPerpendicularVersor(&X_versor,Gsl_random);
	Z_versor=vectorVectorProduct(&X_versor,&Y_versor);
	
	
	
	double *q_local_norm=calloc(ncolloids,sizeof(double));
	bilista *list_solid=bilistaGet(ncolloids);
	clusters *cluster_solid=getClusters(ncolloids);
	int *num_coherent=calloc(ncolloids,sizeof(int));
	int *solidparticle_to_cluster=calloc(ncolloids,sizeof(int));
	interactionmap *coherent_map=createInteractionMap(ncolloids,maxneighbours);
	interactionmap *ime=op->ime;
	
	//resetOP(op,ncolloids,1);
	sph_ws Legendre_workspace;
	double *Legendre_associated_polynomials;
	sph_ws_init(&Legendre_workspace,12);
	Legendre_associated_polynomials=calloc(12+1,sizeof(double));
		
	
	//calculate_qlm_maxneighbours_fast(op,pos,box,range,maxneighbours,cells,X_versor,Y_versor,Z_versor,&Legendre_workspace,Legendre_associated_polynomials);
	calculate_qlm_maxneighbours(op,pos,box,range,maxneighbours,cells,X_versor,Y_versor,Z_versor);
	calculate_Qlm(op,ncolloids);
	getCoherentParticles(12,op->Q,op,ncolloids,threshold,coherent_map,num_coherent,q_local_norm);
	double *Q4map=get_OpL('Q',4,ncolloids,op);
	double *W4map=get_OpL('W',4,ncolloids,op);
	double *Q12map=get_OpL('Q',12,ncolloids,op);
	
	// CLUSTERS
	//bilistaReset(list_solid,ncolloids);
	//resetClusters(cluster_solid,ncolloids);
	
	int particle1,particle2;
	int num_solid=0;
	int max_size=0;
	int root_max_size=0;
	int a_solid=-1;
	for (particle1=0;particle1<ncolloids;particle1++)
	{
		
		
		if  ((num_coherent[particle1]>=numconnections) )
		{
			bilistaInsert(list_solid,particle1);
			a_solid=particle1;
			num_solid++;
			
			addNode(particle1,cluster_solid);
			
			// USUAL DEFINITION
			int j;
			for (j=0;j<ime->howmany[particle1];j++)
			{
				particle2=ime->with[particle1][j];
			
			
			// SAIKA VOIVOD definition
// 			int j;
// 			for (j=0;j<coherent_map->howmany[particle1];j++)
// 			{
// 				particle2=coherent_map->with[particle1][j];
// 				
				if ((num_coherent[particle2]>=numconnections)  )
				{
					
					addNode(particle2,cluster_solid);
					
					int size=addBond(particle1,particle2,cluster_solid);
					
					if (size>max_size)
					{
						max_size=size;
						root_max_size=findRoot(cluster_solid,particle1);
					}
					
				}
			}
		}
	}
	
	if ( (max_size==0) && (num_solid>0) )
	{
		root_max_size=a_solid;
		max_size=1;
	}
	
	// calcoliamo la cluster size distribution
	
	clusterdistribution *csd=calloc(num_solid,sizeof(clusterdistribution));
	int num_clusters=0;
	int i;
	for (i=0;i<ncolloids;i++)
		solidparticle_to_cluster[i]=-1;
	
	if (num_solid>0)
	{
		// partiamo da root_max_size in modo tale che il cluster piu' grande avra' id=0
		i=root_max_size;
		
#ifdef DEBUG
		assert(isRoot(cluster_solid,i)==1);
#endif
		
		do
		{
			int root=findRoot(cluster_solid,i);
			
			int j=0;
			while ((j<num_clusters) && (csd[j].root!=root))
				j++;
			
			solidparticle_to_cluster[i]=j;
			
			if (j==num_clusters)
			{
				// trovato un nuovo root
				csd[num_clusters].id=num_clusters;
				csd[num_clusters].root=root;
				csd[num_clusters].num_particles=1;
				csd[num_clusters].particles=calloc(max_size,sizeof(int));
				csd[num_clusters].particles[0]=i;
				num_clusters++;
			}
			else
			{
				// usiamo un root gia' allocato
				csd[j].particles[(csd[j].num_particles)++]=i;
			}
			
			
			i=list_solid[i].next;
			
			if (i==-1)
				i=list_solid[-1].next;
		}
		while (i!=root_max_size);
	}
	
	// calcoliamo il cm dei singoli cluster
	
	compactcluster c;
	c.label=calloc(max_size,sizeof(int));
	c.whoaddedme=calloc(max_size,sizeof(int));
	c.num=0;
	int *ordered_cluster_labels=calloc(max_size,sizeof(int));
	
	for (i=0;i<num_clusters;i++)
	{
		// aggiungiamo la prima particella
		c.num=1;
		c.label[0]=csd[i].particles[0];
		c.whoaddedme[0]=-1;
		ordered_cluster_labels[0]=csd[i].particles[0];
		int current_cluster_particle=0;
		
		vector com;
		com.x=0.;
		com.y=0.;
		com.z=0.;
		
		while (current_cluster_particle<c.num)
		{
			int label=c.label[current_cluster_particle];
			
			int near_particle=c.whoaddedme[current_cluster_particle];
			
			vector *newpos=pos+label;
			
			if (near_particle!=-1)
			{
				pbcNearestImage(newpos,pos+near_particle,box);
			}
			
			com.x+=newpos->x;
			com.y+=newpos->y;
			com.z+=newpos->z;
			
			// guardiamo ciascun vicino
			int j,neighbour;
			for (j=0;j<csd[i].num_particles;j++)
			{
				neighbour=csd[i].particles[j];
				
				vector olddist,dist;
				olddist.x=pos[label].x-pos[neighbour].x;
				olddist.y=pos[label].y-pos[neighbour].y;
				olddist.z=pos[label].z-pos[neighbour].z;
				
				olddist.x-=rint(olddist.x);
				olddist.y-=rint(olddist.y);
				olddist.z-=rint(olddist.z);
				
				dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
				dist.y=box[3]*olddist.y+box[4]*olddist.z;
				dist.z=box[5]*olddist.z;
				
				if (SQR(dist.x)+SQR(dist.y)+SQR(dist.z)<SQR(range))
				{
					/* if neighbour is not already part of the cluster */
					if (bsearch(&neighbour,ordered_cluster_labels,c.num,sizeof(int),&compareInt)==NULL)
					{
						c.label[c.num]=neighbour;
						c.whoaddedme[c.num]=label;
						insertionSort(ordered_cluster_labels,&(c.num),neighbour);
					}
				}
			}
			current_cluster_particle++;
			
		}
#ifdef DEBUG
		assert(c.num==csd[i].num_particles);
#endif
		
		com.x/=(double)c.num;
		com.y/=(double)c.num;
		com.z/=(double)c.num;
		
		csd[i].cm=com;
	}
	
	
	
	
	for (i=0;i<ncolloids;i++)
	{
		int particle=i;
		
		
		vector dist,olddist;
		double distnorm=-1.;
		int cluster_id=solidparticle_to_cluster[particle];
		int code;
		
		if ( bilistaIsIn(list_solid,particle)==1 )
		{
			
			//code=0;
			if (Q4map[particle]<limit_Q4_clathrate)
			{
				code=CLATRATO_CODE;
			}
			else if (Q4map[particle]<limit_Q4_supercooled)
			{
				code=II_CODE;
			}
			else if (Q4map[particle]<limit_Q4_t12) 
			{
				code=T12_CODE;
			}
			else if (W4map[particle]>0)
			{
				code=IH_CODE;
			}
			else
			{
				code=DC_CODE;
			}
			
			// solid particle
			olddist.x=pos[particle].x-csd[cluster_id].cm.x;
			olddist.y=pos[particle].y-csd[cluster_id].cm.y;
			olddist.z=pos[particle].z-csd[cluster_id].cm.z;
			
			olddist.x-=rint(olddist.x);
			olddist.y-=rint(olddist.y);
			olddist.z-=rint(olddist.z);
			
			dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
			dist.y=box[3]*olddist.y+box[4]*olddist.z;
			dist.z=box[5]*olddist.z;
			
			
			distnorm=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));
		}
		else
		{
			code=LIQUID_CODE;
			// liquid particle
			olddist.x=pos[particle].x-csd[0].cm.x;
			olddist.y=pos[particle].y-csd[0].cm.y;
			olddist.z=pos[particle].z-csd[0].cm.z;
			
			olddist.x-=rint(olddist.x);
			olddist.y-=rint(olddist.y);
			olddist.z-=rint(olddist.z);
			
			dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
			dist.y=box[3]*olddist.y+box[4]*olddist.z;
			dist.z=box[5]*olddist.z;
			
			distnorm=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));
		}
		
		// calcoliamo il parametro d'ordine tetraedrico
		double tetrahedral=0.;
		int k,j;
		int howmany=(ime->howmany[particle]<4 ? ime->howmany[particle] : 4 );
		
		for (j=0;j<howmany;j++)
		{
			int particle_j=ime->with[particle][j];
			
			for (k=j+1;k<howmany;k++)
			{
				int particle_k=ime->with[particle][k];
				
				vector dist_j,dist_k;
				vector olddist_j,olddist_k;
				
				olddist_j.x=pos[particle_j].x-pos[particle].x;
				olddist_j.y=pos[particle_j].y-pos[particle].y;
				olddist_j.z=pos[particle_j].z-pos[particle].z;
				
				olddist_j.x-=rint(olddist_j.x);
				olddist_j.y-=rint(olddist_j.y);
				olddist_j.z-=rint(olddist_j.z);
				
				dist_j.x=box[0]*olddist_j.x+box[1]*olddist_j.y+box[2]*olddist_j.z;
				dist_j.y=box[3]*olddist_j.y+box[4]*olddist_j.z;
				dist_j.z=box[5]*olddist_j.z;
				
				
				double distnorm_j=sqrt(SQR(dist_j.x)+SQR(dist_j.y)+SQR(dist_j.z));
				
				olddist_k.x=pos[particle_k].x-pos[particle].x;
				olddist_k.y=pos[particle_k].y-pos[particle].y;
				olddist_k.z=pos[particle_k].z-pos[particle].z;
				
				olddist_k.x-=rint(olddist_k.x);
				olddist_k.y-=rint(olddist_k.y);
				olddist_k.z-=rint(olddist_k.z);
				
				dist_k.x=box[0]*olddist_k.x+box[1]*olddist_k.y+box[2]*olddist_k.z;
				dist_k.y=box[3]*olddist_k.y+box[4]*olddist_k.z;
				dist_k.z=box[5]*olddist_k.z;
				
				double distnorm_k=sqrt(SQR(dist_k.x)+SQR(dist_k.y)+SQR(dist_k.z));
				
				double costeta=(dist_j.x*dist_k.x+dist_j.y*dist_k.y+dist_j.z*dist_k.z)/(distnorm_j*distnorm_k);
				
				tetrahedral+=SQR(costeta+1./3.);
			}
		}
		
		tetrahedral=1-9.*tetrahedral/(2.*howmany*(howmany-1.));
	
		
		printf("%d %d %lf %lf %lf %lf %lf\n",code,cluster_id,distnorm,Q4map[i],W4map[i],Q12map[i],tetrahedral);
		
	}
	
	// test di coerenza;
	int tot=0;
	for (i=0;i<num_clusters;i++)
	{
		tot+=csd[i].num_particles;
	}
	assert(tot==num_solid);
	
	
	
	
	
	free(csd);
	free(c.label);
	free(c.whoaddedme);
	free(ordered_cluster_labels);
	
	
	freeOP(op,2);
	free(q_local_norm);
	freeList(cells);
	free(pos);
	
	
	return 0;
}


void readPositionsOXDNA(char *_input_name,vector *pos,steps *time,int *numparticles,double NOBox[],double INOBox[])
{
	FILE *ifile=fopen(_input_name,"r");
	
	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}
	
	int state=0;
	
	char line[MAX_LINE_LENGTH]="";
	
	// HEADER: first line
	
	getLine(line,ifile);
	sscanf(line,"t = %lld\n",time);

	getLine(line,ifile);
	sscanf(line,"b = %lf %lf %lf\n",NOBox+0,NOBox+3,NOBox+5);
	NOBox[1]=0.;
	NOBox[2]=0.;
	NOBox[4]=0.;

	getLine(line,ifile);
	
	// check
	int xinversion=1;
	int yinversion=1;
	int zinversion=1;

	int ninversions=0;
	if (NOBox[0]<0)
	{
		xinversion=-1;
		NOBox[0]*=-1;
		ninversions++;
	}
	if (NOBox[3]<0)
	{
		yinversion=-1;
		NOBox[1]*=-1;
		NOBox[3]*=-1;
		ninversions++;
	}
	if (NOBox[5]<0)
	{
		zinversion=-1;
		NOBox[2]*=-1;
		NOBox[4]*=-1;
		NOBox[5]*=-1;
		ninversions++;
	}
	
	assert(ninversions%2==0);
	
	
	INOBox[0]=1./NOBox[0];
	INOBox[1]=-NOBox[1]/(NOBox[0]*NOBox[3]);
	INOBox[2]=(NOBox[1]*NOBox[4])/(NOBox[0]*NOBox[3]*NOBox[5])-NOBox[2]/(NOBox[0]*NOBox[5]);
	INOBox[3]=1./NOBox[3];
	INOBox[4]=-NOBox[4]/(NOBox[3]*NOBox[5]);
	INOBox[5]=1./NOBox[5];
	
	
	int np=0;
	while (getLine(line,ifile)>0)
		np++;
	
	fclose(ifile);	

	ifile=fopen(_input_name,"r");
	getLine(line,ifile);
	getLine(line,ifile);
	getLine(line,ifile);


	int i;

	*numparticles=np;
	
	for (i=0;i<*numparticles;i++)
	{
		vector p;
		getLine(line,ifile);
		state=sscanf(line,"%lf %lf %lf %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f\n",&(p.x),&(p.y),&(p.z));

		pos[i].x=(INOBox[0]*p.x+INOBox[1]*p.y+INOBox[2]*p.z);
		pos[i].y=(INOBox[3]*p.y+INOBox[4]*p.z);
		pos[i].z=(INOBox[5]*p.z);
		
		if (state!=3)
		{
			logPrint("Error while reading '%s' file\n",_input_name);
			exit(1);
		}
	}
	
	fclose(ifile);
}


int getNumberParticlesOXDNA(char *_input_name)
{
	FILE *ifile=fopen(_input_name,"r");
	
	if (ifile==NULL)
	{
		logPrint("Error: can't open initial conditions file '%s'\n",_input_name);
		exit(1);
	}
	
	
	char line[MAX_LINE_LENGTH]="";
	
	// HEADER: first line
	
	getLine(line,ifile);
	getLine(line,ifile);
	getLine(line,ifile);
	
	
	int np=0;
	while (getLine(line,ifile)>0)
		np++;
	
	fclose(ifile);

	return np;

}
