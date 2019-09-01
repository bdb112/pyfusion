# first run  bdb/bdbs_MDSplus_W7X/plot_lissa_smoothed [2.8,2.802] 180911024 100 200 100 1.003 -4
# slow - about 20 -40 secs for 20k points  - also need to remove extra CRs
csvfile=open('20180911024.csv','w')
wr = csv.writer(csvfile)
wr.writerow(['tm','v1','v2','diffamp'])
wr.writerows(np.array([tm, v1,v2,difamp]).T.round(7))
csvfile.close
