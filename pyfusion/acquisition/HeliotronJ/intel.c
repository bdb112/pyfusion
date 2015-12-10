#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* #include <float.h> */
#include <string.h>

extern int *dtptr;
extern unsigned char *btptr;

void little2big(int idata, int adcbit);
void
little2big(idata, adcbit)
int idata, adcbit;
{
int i,j,k, c;
unsigned char s[6];
        switch(adcbit) {
                case 1 :
                        for(i=0; i<idata; ++i) {
                                j = i;
                                        s[0]= *(btptr+j);
                                        s[1] = '\0'; s[2] = '\0';s[3] = '\0';
                                memcpy(dtptr+i, &s[0], 4);
                        }
                                break;
                case 2:
                        for(i=0; i<idata; ++i) {
                                j = 2*i;
                                for (k = 0; k < 2; ++k)
                                        s[k]= *(btptr+j+k);
                                        s[2] = '\0';s[3] = '\0';
                                memcpy(dtptr+i, &s[0], 4);
                        }
                                break;
                default :
                        for(i=0; i<idata; ++i) {
                                j = 4*i;
                                for (k = 0; k < 4; ++k)
                                        s[k]= *(btptr+j+k);
                                memcpy(dtptr+i, &s[0], 4);
                        }
                                break;
        }

}

