#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <float.h>
#include <string.h>   // bdb added
//  two warnings remain like
// ./libfdata.c:480:27: warning: cast from pointer to integer of different size [-Wpointer-to-int-cast]    *(buff+j)=(char)NULL;


#include "helgra.h"

#define HDISKFILE "/HDISK"
#define HDISKLIST "/data/HDISK.list"
#define DEBUG

typedef struct {
        char dir_name[5];
        char lf1;
        char start_shot[5];
        char lf2;
        char end_shot[5];
        char lf3;
        char lf4;
} Hdisklist;

typedef struct {
        char shot_data[5];
        char dummy1;
        char date_data[6];
        char dummy2;
} hdisk_data;

char Data_Dir[128];
char Plot_Item[128];
char *Data_HD[128];


Shot_Param shotparam;

tSampling *head;
tSampling2 *head2;
int *dtptr;
unsigned char *btptr;

void    make_data_dir();
void strip_space(char *buff);
void strip_space2(char *buff);
char    *fGets(char *s, int n, FILE *iop);

int     search_hdisk(int *findno); 
void	getdata_();
extern void little2big(int idata, int adcbit);

/*
void
main(){
char signalnm[16];
char modulenm[16];
char paneldt[100];
char datascl[21];
char dataunit[21];
char stime[6];
char sdate[9];

float dataary[8192];
float tary[8192];
float tsamp, tdelay, ampgain;
float sc_max, sc_min, bitzero;

int	iadc_bit, iampfilt, ishotno;
int	ibhv, ibta, ibtb, ibav, ibiv;
int	ich, isample, iswch;
int	ierror;
int	i;
strcpy(signalnm,"CIIIMONITOR");
ishotno = 884;
iswch = 0;

getdata_(signalnm, modulenm, paneldt, datascl, dataunit, sdate, stime,\
	dataary, tary, &tsamp, &tdelay, &ampgain, \
	&sc_max, &sc_min, &bitzero,\
	&iadc_bit, &iampfilt, &ishotno, &ibhv, &ibta, &ibtb, &ibav, &ibiv, \
	&ich, &isample, &iswch, &ierror);

	printf("signalnm %s\n", signalnm);
	printf("modulenm %s\n", modulenm);
	printf("paneldt %s\n", paneldt);
	printf("datascl %s\n", datascl);
	printf("dataunit %s\n", dataunit);
	printf("sdate %s\n", sdate);
	printf("stime %s\n", stime);
	printf("tdamp %f\n", tsamp);
	printf("tdelay %f\n", tdelay);
	printf("ampgain %f\n", ampgain);
	printf("sc_max %f\n", sc_max);
	printf("sc_min %f\n", sc_min);
	printf("bitzero %f\n", bitzero);
	printf("iampfilt %d\n", iampfilt);
	printf("ishotno %d\n", ishotno);
	printf("ibhv %d\n", ibhv);
	printf("ibta %d\n", ibta);
	printf("ibtb %d\n", ibtb);
	printf("ibav %d\n", ibav);
	printf("ibiv %d\n", ibiv);
	printf("ich %d\n", ich);
	printf("isample %d\n", isample);
	printf("iadc_bit %d\n", iadc_bit);
	printf("ierror %d\n", ierror);

	for (i=0; i<isample; i++) printf("t %f signal %f\n", tary[i], dataary[i]);

}
*/

void
getdata_(signalnm, modulenm, paneldt, datascl, dataunit,  sdate, stime,\
	dataary, tary, tsamp, tdelay, ampgain, \
	sc_max, sc_min, bitzero,\
	iadc_bit, iampfilt, ishotno, ibhv, ibta, ibtb, ibav, ibiv, \
	ich, isample, iswch, ierror)
char *signalnm, *modulenm, *paneldt, *datascl, *dataunit, *sdate, *stime;
float	*dataary, *tary, *tsamp, *tdelay, *ampgain;
float	*sc_max, *sc_min, *bitzero;
int	*iadc_bit,*iampfilt, *ishotno;
int	*ibhv, *ibta, *ibtb, *ibav, *ibiv;
int	*ich, *isample, *iswch;
int	*ierror;
{

	char buff[129];
	float bitmax, r12;
	char *p;
	int i;
        int findno;

	sprintf(shotparam.shot_no, "%05d", *ishotno);

	if(search_hdisk(&findno)) {
		*ierror = 1;
		return;
	}

	make_data_dir();
	strip_space2(signalnm);
	if(Data_File_Read(signalnm, isample, findno)) {
		*ierror = 1;
		return;
	} else *ierror = 0;

/* shot number */
	strcpy(buff, "     ");
//        sprintf(buff, "%5s", &(head->pcdata[6]));
        memcpy(buff,&(head->pcdata[6]),5);
	if(*ishotno != atoi(buff)) *ierror = 2;

/* date */
        memcpy(sdate, head->date, 8);
	sdate[8] = '\0';
     
/* time */
        memcpy(stime, &(head->pcdata[0]), 5);
	stime[5] = '\0';
		
/* H+V(%) */
	strcpy(buff, "    ");
        memcpy(buff, &(head->pcdata[12]), 4);
	*ibhv = atoi(buff);
     
/* TA(%) */
	strcpy(buff, "    ");
        memcpy(buff, &(head->pcdata[32]), 4);
	*ibta = atoi(buff);
     
/* TB(%) */
	strcpy(buff, "    ");
        memcpy(buff, &(head->pcdata[17]), 4);
	*ibtb = atoi(buff);
     
/* AV(%) */
	strcpy(buff, "    ");
        memcpy(buff, &(head->pcdata[22]), 4);
	*ibav = atoi(buff);
     
/* IH(%) */
	strcpy(buff, "    ");
        memcpy(buff, &(head->pcdata[27]), 4);
	*ibiv = atoi(buff);

/* tdelay and tsamp */

                 memset(buff, ' ', 128); 
                 memcpy(buff, &head2->d_time[0], 3);
                 *tdelay = (float) atoi(buff);
                 memset(buff, ' ', 128); 
                 memcpy(buff, &head2->s_time_tm[0], 3);
                 *tsamp = (float) atoi(buff);
                 if(head2->s_time_dm == '0')
                              *tsamp = *tsamp/1000.0;
                 else if(head2->s_time_dm == '2')
                              *tsamp = *tsamp/1000000.0;


                 if(*ishotno > 888 || *ishotno < 393) {
                        memset(buff, ' ', 128);
                        memcpy(buff, &head->preDC[0], 7);
                        *tdelay -= *tsamp * atoi(buff) ;

                 } else {

                         if(strncmp(&head2->modulename[0], "8210", 4) == 0) {
                                        *tdelay -=  *tsamp * 1024.0 ;
                         }
                 }

/* sc_max, sc_min, iadc_bit, bitzero */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->adc_fs[0], 3);
           *sc_max = (float) atoi(buff);
           if(*sc_max==50.0) *sc_max = 0.5;
           memset(buff, ' ', 128);
           memcpy(buff, &head2->adc_fs[3], 3);
           *sc_min = (float) atoi(buff);
           if(*sc_min==-50.0) *sc_min = -0.5;
           memset(buff, ' ', 128);
           memcpy(buff, &head2->adc_fs[6], 2);
           *iadc_bit = atoi(buff);
           bitmax = pow(2, *iadc_bit);

           if((*sc_min) * (*sc_max) < 0) *bitzero = bitmax/2;
           else *bitzero = 0;

/* modulenm */
           memcpy(modulenm, &head2->modulename[0], 15);
		modulenm[15] = '\0';

/* paneldt */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->panel_sel[0], 99);
		memcpy(paneldt, buff, 99);
		paneldt[99] = '\0';

/* datascl */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->scale[0], 20);
		memcpy(datascl, buff, 20);
		datascl[20] = '\0';

/* dataunit */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->unit[0], 20);
		memcpy(dataunit, buff, 20);
		dataunit[20] = '\0';

/* channel number */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->ch_no[0], 2);
		*ich = atoi(buff);

/* ampgain */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->amp_gain[0], 4);
		*ampgain = atof(buff);
		if( *ampgain < 0.0) *ampgain = - *ampgain/100.0;

/* ampfilter */
           memset(buff, ' ', 128);
           memcpy(buff, &head2->amp_filter[0], 3);
		*iampfilt = atoi(buff);

	if(*iswch == 1)
          r12 = ((*sc_max)- (*sc_min))/bitmax/(*ampgain);
	else
	 r12 = 1.0;

        for (i = 0; i < *isample; i++) {
		*(dataary+i) = (*(dtptr+i)-(*bitzero)) * r12;
		*(tary+i) = *tdelay + (float)i * (*tsamp);
	}
        free(head);
        free(head2); 
        free(dtptr);


		return;
}

void
make_data_dir()
{
char buff[130];

        
        strcpy(Data_Dir , "/");
        strncat(Data_Dir , shotparam.date,6); 
        strcat(Data_Dir , "/");
        strncat(Data_Dir , shotparam.shot_no,5);
        strcat(Data_Dir , "/");
}               
int
search_hdisk(findno)
int *findno;
{
           
hdisk_data read_hdisk;
Hdisklist  hdisklist;
Hdisklist  sdisklist[100];
FILE *fp;        
char buff1[10];  
char hdiskfile[128];
int     i,l1,len;
int     startshot, endshot, sshot;

        *findno = 0;
        for(i=0;i<128;i++)    
                Data_HD[i] = (char *)malloc(10);
		// HDISK.list must be in fixed format 
        if((fp = fopen(HDISKLIST, "r")) == NULL) {
                printf("no hdisk list\n");
                return 1;
        }
           
		// Add the a phantom local directory at the head of the list, so it looks there first for local caching
		l1 = 0;
		memcpy(&sdisklist[l1].dir_name[0], "/hdd2 54000 99999   " , sizeof(Hdisklist));              
                sdisklist[l1].lf1 = '\0';
                sdisklist[l1].lf2 = '\0';
                sdisklist[l1].lf3 = '\0';

				l1 = 1;   // Now start at 1 instead of 0
        for (;;) {
        if(fread(&hdisklist, 1, sizeof(Hdisklist), fp) < 1) {
                break;
        } else {
                memcpy(&sdisklist[l1].dir_name[0], &hdisklist.dir_name[0], sizeof(Hdisklist));              
                sdisklist[l1].lf1 = '\0';
                sdisklist[l1].lf2 = '\0';
                sdisklist[l1].lf3 = '\0';
                l1++;
        }
        }
        fclose(fp); 
                
                
        for (i = 0; i<l1; i++) {
                strcpy(hdiskfile, sdisklist[i].dir_name);
                strcat(hdiskfile, HDISKFILE);
                startshot = atoi(sdisklist[i].start_shot);
                endshot = atoi(sdisklist[i].end_shot);
                sshot = atoi(shotparam.shot_no);
#ifdef DEBUG
printf("%d %s\n",i, hdiskfile);
printf("start shot %d endshot %d search shot %d\n",startshot, endshot, sshot);
#endif
                if(sshot>=startshot && sshot <=endshot) {
                        if((fp = fopen(hdiskfile, "r")) == NULL)
                                continue;
                        for( ;; ) {
                                if(fread(&(read_hdisk), 1, sizeof(hdisk_data), fp) == 0) {
                                        fclose(fp);
                                        break;
                                }
                                if(strncmp(&(read_hdisk.shot_data[0]), shotparam.shot_no,5) == 0) {    
                                        fclose(fp);
                                        strncpy(shotparam.date, read_hdisk.date_data,6);
                                        strcpy(Data_HD[*findno], sdisklist[i].dir_name);
                                        *findno +=1;
                                        break;
                                }
                        }
                }
        }        
        if(*findno > 0) {
		  #ifdef DEBUG 
		  printf("IFOUND>0: found! \n");		  
		  #endif
		  return 0;
		}
        else {
		  #ifdef DEBUG 
		  printf("IFOUND=0: not found\n");
		  #endif
		  return 1;
		}
}

int
Data_File_Read(signalnm, isample, findno)
char *signalnm;
int	*isample;
int     findno; 
{             
int i,i1;  
FILE *fp;               
char buff[130];
char buff1[17];
int idata;
int preDC;              
int shotno;
float s_time;    
int data_exist;
int adcbit;

 
        shotno = atoi(shotparam.shot_no); 

        head = (tSampling *)malloc( sizeof(tSampling));
        head2 = (tSampling2 *)malloc( sizeof(tSampling2));
                                
        for (i = 0; i<findno; i++) {
                memset(buff, ' ', 128);
                memset(buff1, ' ', 15);
                strcpy(buff, Data_HD[i]);
                strcat(buff, Data_Dir);

                memcpy(&buff1[0], signalnm, 15);
                strip_space(&buff1[0]);
                strcat(buff, buff1);
#ifdef DEBUG 
		printf("trying %s\n", buff);
#endif
                if( (   fp = fopen(buff,"r")) != NULL)
                        break;
        }

        for(i1=0;i1<128;i1++) free(Data_HD[i1]);

        if(i == findno) {
                free(head);     
                free(head2);    
                return 1;
        }

        fread(head, 1, sizeof(tSampling), fp);
        fread(head2, 1, sizeof(tSampling2), fp);

        idata = atoi(head2->dataword);
        if(shotno < 889 && shotno > 392) {
             if(strncmp(&head2->modulename[0], "8210", 4) == 0) {
                          idata = 8192;
             }
        }               
        memcpy(&buff1[0], &head2->adc_fs[6],2);
        buff1[2]='\0';
        adcbit = atoi(buff1);

        dtptr = (int *)malloc(idata * sizeof(int));
        btptr = (char *)malloc(idata * sizeof(int)+1);

        if(shotno < 5316 && shotno > 300) {
               fread(btptr,1, idata * sizeof(int), fp);
               adcbit = 4;
        } else {
              if(adcbit <9) adcbit = 1;
                else if(adcbit <17) adcbit = 2;
                else adcbit = 4;
             fread(btptr,1, idata * adcbit, fp);
         }

        fclose(fp);     
	*isample = idata;

        little2big(idata,adcbit);
        free(btptr);
	return 0;

 
}   


void
strip_space(buff)
char *buff;
{
int i,j;
        i = strlen(buff);

        if(i == 0) return;

        for(j = i-1; j>0; j--) {
                if(*(buff+j) != ' ')
                        break;
        }
        
        if(j < i) 
                *(buff+j+1)=(char)NULL;

}       
void
strip_space2(buff)
char *buff;
{
int i,j;
        i = strlen(buff);

        if(i == 0) return;

        for(j = 0; j<i; j++) {
                if(*(buff+j) == ' ')
                        break;
        }
        
        if(j < i) 
                *(buff+j)=(char)NULL;

}       
char *fGets(char *s, int n, FILE *iop)
{
        register int c;
        register char *cs;

        cs = s;
        while (--n > 0 && (c = getc(iop)) != EOF) {
                if( c == '\n' )
                        break;
                *cs++ = c;
        }
        *cs = '\0';
        return (c == EOF && cs == s) ? NULL : s;
}
