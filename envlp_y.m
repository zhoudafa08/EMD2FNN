function [envx, envm]=envlp_y(xx,method)
%   [envx,envm]=envlp(xx): Envelop of maximum and minimum points of improved data xx(n,k)
%      by extending two waves at each end.
%      where k: the sifted components; n: length of the time series.
%  Input:
%      x: data
%  Output:
%      envx: envelop of maximum points;
%      envm: envelop of minimum points.
%  method='pchip', if use pchip to interplate the data
%  method='spline' or others, if use cubic spline to interplate the data

%  
%  D. XIANG   06-10-2002  extend both ends with 2 waves,
%             then use cubic spline to find the envelop for
%             Max points 
%  At the Johns Hopkins University

%%----------------------------------------------------------------------------------
%  Lihua Yang remarks that in Sept. 2011 in Sun Yat-sen University
%     xx is a matrix of $n\times k$, each column of it is a time series of length n.  
%      k is a natural number.
%     The output envx and envm are two matrice of the same size as xx. Each column of them is
%     the envelopes for the corresponding row of xx.
%
%%----------------------------------------------------------------------------------


[n,kpt]=size(xx);
yy=zeros(n,kpt);

for j=1:kpt

% -----------------------find extrima--------------------------------
  
 
    maxp=locmax(xx);
    maxx=maxp(:,1);
    maxy=maxp(:,2);
    clear maxp;
    maxp=locmin(xx);
    minx=maxp(:,1);
    miny=maxp(:,2);
  
    
%-------------------Treat the head ----------------------------------
    n_mx=maxx(1);
    n_mn=minx(1);
    e_mx=maxx(end);
    e_mn=minx(end);
    
    if (n_mx==-1) | (n_mn==-1) % no max or min
        disp('At least one component does not have Max or Min!');
        break;
    elseif n_mn<n_mx,
        dlt=n_mx-n_mn;
    else
        dlt=n_mn-n_mx;
    end
    
    while(maxx(1)>0|minx(1)>0)
        maxx=[maxx(1)-2*dlt; maxx];
        maxy=[maxy(1);maxy];
        maxx=[maxx(1)-2*dlt; maxx];
	    maxy=[maxy(1);maxy];
        minx=[minx(1)-2*dlt; minx];
        miny=[miny(1);miny];
        minx=[minx(1)-2*dlt; minx];
        miny=[miny(1);miny];
    end
    
    %---------------------Treat the tail----------------------------
    if e_mn<e_mx,
        dlt=e_mx-e_mn;
    else
        dlt=e_mn-e_mx;
    end
    
    while(maxx(end)<n|minx(end)<n)
        maxx=[maxx; maxx(end)+2*dlt];
        maxy=[maxy; maxy(end)];
        maxx=[maxx; maxx(end)+2*dlt];
	    maxy=[maxy; maxy(end)];
        minx=[minx; minx(end)+2*dlt];
        miny=[miny; miny(end)];
        minx=[minx; minx(end)+2*dlt];
        miny=[miny; miny(end)];
    end
    
       
    if maxx(1)>0|maxx(end)<n|minx(1)>0|minx(end)<n
        disp('Extending end fail!');
        break;
    end
    
    mx=(maxx(1):maxx(end))';
    nx=(minx(1):minx(end))';
    if strcmp(method,'pchip')
        my=pchip(maxx, maxy, mx);
        ny=pchip(minx, miny, nx);
    else
        my=spline(maxx, maxy, mx);
        ny=spline(minx, miny, nx);
    end
	  
    np1=1;
    while(mx(np1)<=0)
        np1=np1+1;
    end
    np2=1;
    while(nx(np2)<=0)
        np2=np2+1;
    end
    
	xx(:,j)=my(np1:np1+n-1);
    yy(:,j)=ny(np2:np2+n-1);
end
%h=x;
envx=xx;
envm=yy;
clear xx yy
