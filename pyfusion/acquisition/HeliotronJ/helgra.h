/************************************************************************/
/*									*/
/*	ModuleName		:	header file			*/
/*	FileName		:	seq_proc.h			*/
/*	Author			:	H.Nishide			*/
/*	Create Date		:	92/01/07			*/
/*	ReCreate Date	:	96/07/22				*/
/*									*/
/************************************************************************/
/****************************************************************
	sampling header table	
****************************************************************/
#define         UCHAR   unsigned char

typedef	struct	{
		UCHAR	date[8];		/* date */
		UCHAR	pcdata[47];		/* data from PC */
		UCHAR	preDC[7];		/* pre-trigger data word */
		UCHAR	lf;
}	tSampling ;

typedef	struct	{
		UCHAR	modulename[15];		/* Module Name	*/
		UCHAR	ch_no[2];		/* channel No.	*/
		UCHAR	dataword[7];		/* DataWord	*/
		UCHAR	s_time_tm[3];		/* SamplingTime(Time) */
		UCHAR	s_time_dm;		/* SamplingTime(Dimm)	*/
		UCHAR	d_time[3];		/* DelayTime		*/
		UCHAR	amp_gain[4];		/* AmpGain		*/
		UCHAR	amp_filter[3];		/* AmpFilter		*/
		UCHAR	panel_sel[99];		/* panel setup value*/
		UCHAR	scale[20];		/* Scale		*/
		UCHAR	unit[20];		/* Unit			*/
		UCHAR	adc_fs[8];		/* Camac ADC F.S	*/
		UCHAR	inf[50];		/* Information		*/
}	tSampling2 ;


typedef struct {
		UCHAR date[7];
		UCHAR shot_no[6];
}	Shot_Param;

/****************************************************************
	function section data value table	
****************************************************************/
	char	*DtTbl[]	=	{
/*		-1    -2   -3    -4    -5     -6      -7      -8	*/
/*		"-1", "Q", "D1", "DC", "CLK", "DCNT", "DCLK", "DFRM", "" 	*/
		"-1   ", "Q    ", "D1   ", "DC   ", "CLK  ", "DCNT ", "DCLK ", "DFRM ", "" 
	} ;

#define		HD_Dir	"/data"
#define		NSHOT_MAX	100
