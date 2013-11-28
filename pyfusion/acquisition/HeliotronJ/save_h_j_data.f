C This program reads Heliotron-J data, and writes it to a file for reading mainly by IDL 
C See read_h_j_file.pro
C Present behaviour - three args on one line, the file name quoted.
C [Old It accepts three arguments on separate lines, 
C the signal name, shot number, and unformatted file name.]
C if the unformatted file name is "nofile", no data is written
C If the unformatted file name is empty, or simply 'f' for the
C  "all on one line case"  then only formatted data are written, 
C otherwise only the data description is written formatted and 
C the time series data unformatted for speed.
C the orderinternal to the raw file is 
C 1/ Rubbish  2/ shot, sig, 3/ SIGNAL::  4/ filname or f 
C 5/ishotno, isample, tdelay, tsamp  6/ time-series array 7/ Namelist
c$ g77 -Lcdata test_h_j_data.f -lfdata
c$ 2013: add option to have no data (to get params easily from idl or python)
c# timing 2013: pchelio12 fresh shot, nofile DIA135 25/sec MP1 6/sec
c$ Timing 2009:  (pchelio fresh shot)
c$ /bin/echo -e 'MP1 33919 "/var/tmp/foo"'|./save_h_j_data |wc -c   0.1sec binary
c$ /bin/echo -e 'MP1 33919 f'|./save_h_j_data |wc -c   0.5-0.7 sec formatted
c$ echo MICRO01 16858 f | a.out > MICRO01.16858
c$ /bin/echo -e "DIA135 \n 16858 \n f"| a.out > DIA135.16858
c$ echo 'MP1 16858 "/var/tmp/fort.8"'| a.out > DIA135.16858
C: to /var/tmp from MP! (250k 4byte samples - 0.1 sec/read
C Note:  if using file names beginning with /, then need to quote as above
c
c from kobayashi/yamamoto
	implicit real (a-h,o-z)
c
	external getdata
c
	parameter (BIGDATA=10 000 000)   ! yes, you can have spaces in fortran statements - crazy but true!
	dimension dataary(BIGDATA),tary(BIGDATA),rdata(BIGDATA)
	integer iadc_bit,iampfilt,ishotno
	integer ibhv,ibta,ibtb,ibav,ibiv
	integer ich,isample,iswch
	integer ierror
c
	character*16 signalnm,modulenm
	character*100 paneldt,unf_file
	character*21 datascl,dataunit
	character*6 stime
	character*9 sdate
	character*5 cshot
c
		logical fmtted/.false./
c
	namelist/DATAPARM/ signalnm,modulenm,paneldt,datascl,dataunit,sdate,
     &	stime,tsamp,tdelay,ampgain,sc_max,sc_min,bitzero,
     &	iadc_bit,iampfile,ishotno,ibhv,ibta,ibtb,ibav,ibiv,
     &	ich,isample,iswch,ierror,unf_file
c
	signalnm='DIA135'
	write(6,*) 'signalnm,shotno free format'
	read(5,*) signalnm,ishotno,unf_file
	write(6,*) signalnm,ishotno,unf_file
	fmtted = ((len_trim(unf_file) .le. 1) .and. (unf_file .eq. "f")) ! sometimes null strings are too hard
c	print *, len_trim(unf_file), fmtted
c	ishotno=7985
	iswch=1
c
	call getdata(signalnm,modulenm,paneldt,datascl,dataunit,sdate,
     &	stime,dataary,tary,tsamp,tdelay,ampgain,sc_max,sc_min,bitzero,
     &	iadc_bit,iampfile,ishotno,ibhv,ibta,ibtb,ibav,ibiv,
     &	ich,isample,iswch,ierror)
c
	if(ierror.ne.0) then 
		write(6,*) 'read error'
		call exit(ierror)  ! can't find data -> 1 I think
		end if
c
	do 200 i=1,isample
c	 dataary(i)=1.0*dataary(i)
	 rdata(i)=dataary(i)
 200	continue
c
c
c	call offset(rdata)
c
c This signals the beginning of the data - first the signal name, then shot, sample number, tdelay and tsamp
C probably should put this in a temporary area? - the name can be found from the formatted file
c		   unf_file="fort.8"

		if (fmtted)  then
		   write (6,"('SIGNAL::',a20)") signalnm
		   print *, ishotno, isample, tdelay, tsamp
		   write (6,*) (rdata(j),j=1,isample)
		else
		   if (unf_file .ne. 'nofile') then
		      OPEN(8, FILE=unf_file, FORM="unformatted") 
		      write (8) signalnm
		      write (8) ishotno, isample, tdelay, tsamp
		      write (8) (rdata(j),j=1,isample)
		   endif
		endif
c put the namelist data to a text file for convenience
		write(6,DATAPARM)
		if (0 .eq.1) then
		   write(6,*) 'saving filtering data'
		   write(6,*) '200 micro sec sampling'
		   open(10,file='data/'//cshot//'.diamagf')
		   write(10,*) '"filtering data'
		   do 110 j=1,BIGDATA,40
			  write(10,'(f7.2,1x,1pe10.3)') tary(j),rdata(j)
 110	   continue
		   
		   close(10)
		endif
		
		
		stop
		end
		

c --- subroutines


c --- subroutine for offset
c
	subroutine offset(rdata)
c
	implicit real*8(a-h,o-z)
c
	parameter (BIGDATA=10 000 000)
	real rdata(BIGDATA)
c
	istart=1
	iend=2000
	roffset=0.
	do 100 i=istart,iend
	 roffset=roffset+rdata(i)/2000.
 100	continue
c
	do 200 j=1,BIGDATA
	   rdata(j)=rdata(j)-roffset
 200	continue
c
	return
	end
