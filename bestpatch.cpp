#include<iostream>
#include<cstdio>
#include<cmath>
#include<vector>
using namespace std;

extern "C" void cfun(const void * indatav, int first, int second, int third, int fourth, const void * outdatav, const void* maskp, void * indexp) {
    //void cfun(const double * indata, int first, int second, double * outdata) {
    const double * indata = (double *) indatav;
    const double * outdata = (double *) outdatav;
    const bool * mask = (bool *) maskp;
    double * errors= new double[first]();
    int* index = (int*) indexp;
    int i,x,y;
    // puts("Here we go!");
    // cout<<first<<' '<<second<<' '<<third<<' '<<fourth<<' '<<endl;
    // puts(first);puts(second);puts(third);
    vector<pair<int, int> > indices;

    for(int i=0;i<second*third;i++)
    {
      // for(int j=0;j<third;j++){
        // cout<<mask[i]<<' ';
      // }
    }
    cout<<endl;

    for(int i=0;i<second;i++)
    {
      for(int j=0;j<third;j++){
        if(mask[i*third+j]) indices.push_back(make_pair(i,j));
      }
    }

    // cout<<indices.size();

    // for(vector< pair<int, int> >::iterator i = indices.begin();i!=indices.end();++i)
    // {
    //   cout<< i->first << ' '<< i->second<<endl;
    // }

    for (i = 0; i < first; ++i) {
        x= i*second*third*fourth;

        for(vector< pair<int, int> >::iterator it = indices.begin();it!=indices.end();++it)
        {
          y= (it->first)*third*fourth + (it->second)*fourth;
          for(int l=0;l<fourth;l++)
          {
            // cout<<(i*second*third+j*third+k)<<endl;
            // if(i==1){
            //   cout<<x<<' '<<y<<' '<<x+y+l<<' '<<y+l<<endl;
            // }
            errors[i] += pow((indata[x+y+l] - outdata[y+l]),2);
          }
        }
        // cout<<error<<endl;
        // errors[i] = error;
    }
    double mini = 100000000.0;
    int ind=-1;
    for(int i=0;i<first;i++)
    {
      // cout<<errors[i]<<' ';
      if(mini > errors[i])
      {
        mini = errors[i];
        ind = i;
      }
    }

    vector<int> minimas;

    for(int i=0;i<first;i++)
    {
      // cout<<errors[i]<<' ';
      if(errors[i] == mini)
      {
        minimas.push_back(i);
      }
    }

    *index = ind;

    if(minimas.size() > 1)
    {
      int n =minimas.size();
      double variance[n];
      double size = second*third*fourth;
      double sum,mean;
      for (i=0;i<n;i++)
      {
        sum=0.0;
        x=minimas[i]*second*third*fourth;
        for(int j = 0; j<second;j++)
        {
          y = j*third*fourth;
          for(int k=0;k<third;k++)
          {
            for(int l=0;l<fourth;l++) sum+=indata[x+y+(k*fourth)+l];
          }
        }
        mean = 1.0*sum/size;
        sum=0.0;
        for(int j = 0; j<second;j++)
        {
          y = j*third*fourth;
          for(int k=0;k<third;k++)
          {
            for(int l=0;l<fourth;l++) sum+=pow((indata[x+y+(k*fourth)+l]-mean),2);
          }
        }
        variance[i]=sum;
      }
      sum=10000000.0;
      ind=-1;
      for(i=0;i<n;i++)
      {
        if(variance[i]<sum)
        {
          sum=variance[i];
          ind = i;
        }
      }
      *index = ind;
    }

    // cout<<endl<<errors[*index]<<endl;
    // cout<<*index<<endl;
    // puts("Done!");
}
