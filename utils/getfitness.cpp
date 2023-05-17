4/*#include<bits/stdc++.h>
#include<algorithm>
using namespace std; */
struct seg1
{
	double pos;
	double s;
	double e;
} xsegment[1010],ysegment[1010],r_x[1010],r_y[1010];
//水平线[y坐标 左起点x 右终点x],纵线段[x坐标 低起点y 高终点y]
struct seg2
{
	double sx;
	double sy;
	double ex;
	double ey;
} x45[1010],x135[1010];
//斜线段[起点x坐标，起点y坐标，终点x坐标，终点y坐标]
bool cmp1(struct seg1 a,struct seg1 b)
{
	if(a.pos<b.pos) return true;
	else return false;
}
bool cmp2(struct seg1 a,struct seg1 b)
{
	if(a.s<b.s) return true;
	else return false;
}
double Getfitness(double point_arry[][2],int xes_arry[],int num)
{
	int i,sp,xt=0,yt=0,x45t=0,x135t=0,rxt=0,ryt=0,k;
	double x1,x2,y1,y2,fx,fy,fitness=0,midx,midy,P;
//	printf("%d\n",num);
//	for(i=1;i<=3*(num-1);i++)
//		printf("%d ",xes_arry[i]);
//	printf("\n");
	P=0.5*sqrt(2) ;
	for(i=1; i<num; i++)//n个点遍历n-1次以存数据 
	{//xes_arry[i] 数组内连续三个为一组表示xes_arry [起点序号 终点序号 连接方式] 
		//point_arry[点序号p][0] p点的x坐标;point_arry[点序号p][1]p点的y坐标
		x1=point_arry[xes_arry[i*3-2]][0]; //起点x
		y1=point_arry[xes_arry[i*3-2]][1]; //起点y
		x2=point_arry[xes_arry[i*3-1]][0]; //终点x
		y2=point_arry[xes_arry[i*3-1]][1]; //终点y
		sp=xes_arry[i*3]; //连接方式  sp=0走直角/sp=1斜线 
		fx=fabs(x1-x2);  //水平距离
		fy=fabs(y1-y2);  //垂直距离
//		printf("%lf %lf %lf %lf %d %lf %lf\n",x1,y1,x2,y2,sp,fx,fy);
		
		if(sp==0)//当线段曼哈顿连接时 （此时靠近末端为竖线） 
		{
			xsegment[xt].pos=y1;
			ysegment[yt].pos=x2;
			if(x2<x1)
			{
				xsegment[xt].s=x2; //水平线左起点横坐标
				xsegment[xt++].e=x1; //水平线右终点横坐标
			}
			else
			{
				xsegment[xt].s=x1;
				xsegment[xt++].e=x2;
			}
			if(y2<y1)
			{
				ysegment[yt].s=y2; //垂直线下起点纵坐标
				ysegment[yt++].e=y1; //垂直线上终点纵坐标
			}
			else
			{
				ysegment[yt].s=y1;
				ysegment[yt++].e=y2;
			}
			
		}
		if(sp==1) //当线段的连接方式为非曼哈顿结构时
		{
			if(fx==0||fy==0) //两点共线的情况
			{
				if(fy==0) //水平共线 
				{
					xsegment[xt].pos=y1;
					if(x1>x2)
					{
						xsegment[xt].s=x2; //水平线左起点横坐标
						xsegment[xt++].e=x1; //水平线右终点横坐标
					}
					else
					{
						xsegment[xt].s=x1;
						xsegment[xt++].e=x2;
					}
				}
				else//垂直共线 
				{
					ysegment[yt].pos=x1;
					if(y1>y2)
					{
						ysegment[yt].s=y2; //垂直线下起点纵坐标
						ysegment[yt++].e=y1; //垂直线上终点纵坐标
					}
					else
					{
						ysegment[yt].s=y1;
						ysegment[yt++].e=y2;
					}
				}
			}
			else //两点不共线
			{
				if(fx>fy) //水平间距大于垂直间距的情况
				{
					if(x1<x2)
					{
						xsegment[xt].pos=y1;
						xsegment[xt].s=x1;
						xsegment[xt++].e=x2-fy;
						if(y1<y2) //记录45度线段的起点和终点坐标
						{
							x45[x45t].sx=x2-fy;
							x45[x45t].sy=y1;
							x45[x45t].ex=x2;
							x45[x45t++].ey=y2;
						}
						else  //记录135度线段的起点和终点坐标
						{
							x135[x135t].sx=x2;
							x135[x135t].sy=y2;
							x135[x135t].ex=x2-fy;
							x135[x135t++].ey=y1;
						}
					}
					else
					{
						xsegment[xt].pos=y1;
						xsegment[xt].s=x2+fy;
						xsegment[xt++].e=x1;
						if(y1>y2) //记录45度线段的起点和终点坐标
						{
							x45[x45t].sx=x2;
							x45[x45t].sy=y2;
							x45[x45t].ex=x2+fy;
							x45[x45t++].ey=y1;
						}
						else  //记录135度线段的起点和终点坐标
						{
							x135[x135t].sx=x2+fy;
							x135[x135t].sy=y1;
							x135[x135t].ex=x2;
							x135[x135t++].ey=y2;
						}
					} 
				} 
				else //垂直间距大于或等于水平间距的情况
				{
					if(y1>y2)
					{
						ysegment[yt].pos=x1;
						ysegment[yt].s=y2+fx;
						ysegment[yt++].e=y1;
						if(x2<x1) //记录45度线段的起点和终点坐标
						{
							x45[x45t].sx=x2;
							x45[x45t].sy=y2;
							x45[x45t].ex=x1;
							x45[x45t++].ey=y2+fy;
						}
						else  //记录135度线段的起点和终点坐标
						{
							x135[x135t].sx=x2;
							x135[x135t].sy=y2;
							x135[x135t].ex=x1;
							x135[x135t++].ey=y2+fy;
						}
					}
					else
					{
						ysegment[yt].pos=x1;
						ysegment[yt].s=y1;
						ysegment[yt++].e=y2-fx;
						if(x2>x1) //记录45度线段的起点和终点坐标
						{
							x45[x45t].sx=x1;
							x45[x45t].sy=y2-fy;
							x45[x45t].ex=x2;
							x45[x45t++].ey=y2;
						}
						else  //记录135度线段的起点和终点坐标
						{
							x135[x135t].sx=x1;
							x135[x135t].sy=y2-fy;
							x135[x135t].ex=x2;
							x135[x135t++].ey=y2;
						}
					}
				} 
			}
		} 	
	} 
	for(i=0; i<x45t; i++) //将45度线顺时针旋转45度变为水平，并加入xsegment中
	{
		//以45度线延长线与y轴交点为原点（圆心）旋转  p=0.5*sqrt（2） 
		r_x[rxt].pos=P*(x45[i].sy-x45[i].sx); //旋转后的y坐标
		r_x[rxt].s=P*(x45[i].sx+x45[i].sy);   //旋转后的起点x坐标 
		r_x[rxt++].e=P*(x45[i].ex+x45[i].ey); //旋转后的终点x坐标
	}
		for(i=0; i<x135t; i++) //将135度线顺时针旋转45度变为垂直，并加入ysegment中
	{
		//以中垂线延长线与轴交点为原点（圆心）旋转  p=0.5*sqrt（2） 
		r_y[ryt].pos=P*(x135[i].sx+x135[i].sy);
		r_y[ryt].s=P*(x135[i].sy-x135[i].sx);
		r_y[ryt++].e=P*(x135[i].ey-x135[i].ex);
	}
	/*调整起点在左下（可省）*/ 
	for(i=0; i<xt; i++) //调整水平线段，使得起点在左，终点在右
	{
		if(xsegment[i].s>xsegment[i].e)
		{
			midx=xsegment[i].s;
			xsegment[i].s=xsegment[i].e;
			xsegment[i].e=midx;
		}
	}
	for(i=0; i<yt; i++)//调整垂直线段，使得起点在下，终点在上
	{
		if(ysegment[i].s>ysegment[i].e)
		{
			midy=ysegment[i].s;
			ysegment[i].s=ysegment[i].e;
			ysegment[i].e=midy;
		}
	}
	for(i=0; i<rxt; i++)//调整水平线段，使得起点在左，终点在右
	{
		if(r_x[i].s>r_x[i].e)
		{
			midx=r_x[i].s;
			r_x[i].s=r_x[i].e;
			r_x[i].e=midx;
		}
	}
	for(i=0; i<ryt; i++)//调整垂直线段，使得起点在下，终点在上
	{
		if(r_y[i].s>r_y[i].e)
		{
			midy=r_y[i].s;
			r_y[i].s=r_y[i].e;
			r_y[i].e=midy;
		}
	}
	/*按pos与s排序存储*/
	sort(xsegment,xsegment+xt,cmp1);  //按照第一列pos的升序排列
	sort(ysegment,ysegment+yt,cmp1);
	sort(r_x,r_x+rxt,cmp1);
	sort(r_y,r_y+ryt,cmp1);
	for(i=1; i<xt; i++) //水平线在pos升序的基础上，再按照s升序排列
	{
		k=i-1;
		while(i<xt&&xsegment[i].pos==xsegment[i-1].pos)
			i++;
		sort(xsegment+k,xsegment+i,cmp2);
	}
	for(i=1; i<yt; i++) //垂直线在pos升序的基础上，再按照s升序排列
	{
		k=i-1;
		while(i<yt&&ysegment[i].pos==ysegment[i-1].pos)
			i++;
		sort(ysegment+k,ysegment+i,cmp2);
	}
	for(i=1; i<rxt; i++) //旋转后的水平线在pos升序的基础上，再按照s升序排列
	{
		k=i-1;
		while(i<rxt&&r_x[i].pos==r_x[i-1].pos)
			i++;
		sort(r_x+k,r_x+i,cmp2);
	}
	for(i=1; i<ryt; i++) //旋转后的垂直线在pos升序的基础上，再按照s升序排列
	{
		k=i-1;
		while(i<ryt&&r_y[i].pos==r_y[i-1].pos)
			i++;
		sort(r_y+k,r_y+i,cmp2);
	}
	/*合并重叠部分*/ 
	for(i=1; i<xt; i++) //合并重叠的原水平线
	{
		if(xsegment[i].pos==xsegment[i-1].pos)
			if(xsegment[i].s==xsegment[i-1].s)
			{
				if(xsegment[i].e<xsegment[i-1].e)
					xsegment[i].e=xsegment[i-1].e;
				xsegment[i-1].pos=0;
				xsegment[i-1].s=0;
				xsegment[i-1].e=0;
			}
			else
			{
				if(xsegment[i].s<xsegment[i-1].e)
				{
					xsegment[i].s=xsegment[i-1].s;
					if(xsegment[i].e<xsegment[i-1].e)
						xsegment[i].e=xsegment[i-1].e;
					xsegment[i-1].pos=0;
					xsegment[i-1].s=0;
					xsegment[i-1].e=0;
				}
			}
	}
	for(i=1; i<yt; i++) //合并重叠的原垂直线
	{
		if(ysegment[i].pos==ysegment[i-1].pos)
			if(ysegment[i].s==ysegment[i-1].s)
			{
				if(ysegment[i].e<ysegment[i-1].e)
					ysegment[i].e=ysegment[i-1].e;
				ysegment[i-1].pos=0;
				ysegment[i-1].s=0;
				ysegment[i-1].e=0;
			}
			else
			{
				if(ysegment[i].s<ysegment[i-1].e)
				{
					ysegment[i].s=ysegment[i-1].s;
					if(ysegment[i].e<ysegment[i-1].e)
						ysegment[i].e=ysegment[i-1].e;
					ysegment[i-1].pos=0;
					ysegment[i-1].s=0;
					ysegment[i-1].e=0;
				}
			}
	}
	for(i=1; i<rxt; i++) //合并旋转后重叠的水平线
	{
		if(r_x[i].pos==r_x[i-1].pos)
			if(r_x[i].s==r_x[i-1].s)
			{
				if(r_x[i].e<r_x[i-1].e)
					r_x[i].e=r_x[i-1].e;
				r_x[i-1].pos=0;
				r_x[i-1].s=0;
				r_x[i-1].e=0;
			}
			else
			{
				if(r_x[i].s<r_x[i-1].e)
				{
					r_x[i].s=r_x[i-1].s;
					if(r_x[i].e<r_x[i-1].e)
						r_x[i].e=r_x[i-1].e;
					r_x[i-1].pos=0;
					r_x[i-1].s=0;
					r_x[i-1].e=0;
				}
			}
	}
	for(i=1; i<ryt; i++) //合并旋转后重叠的垂直线
	{
		if(r_y[i].pos==r_y[i-1].pos)
			if(r_y[i].s==r_y[i-1].s)
			{
				if(r_y[i].e<r_y[i-1].e)
					r_y[i].e=r_y[i-1].e;
				r_y[i-1].pos=0;
				r_y[i-1].s=0;
				r_y[i-1].e=0;
			}
			else
			{
				if(r_y[i].s<r_y[i-1].e)
				{
					r_y[i].s=r_y[i-1].s;
					if(r_y[i].e<r_y[i-1].e)
						r_y[i].e=r_y[i-1].e;
					r_y[i-1].pos=0;
					r_y[i-1].s=0;
					r_y[i-1].e=0;
				}
			}
	}
	
	for(i=0; i<xt; i++)//统计所有原水平线的长度 
		fitness+=(xsegment[i].e-xsegment[i].s);
	for(i=0; i<yt; i++)//统计所有原垂直线的长度 
		fitness+=(ysegment[i].e-ysegment[i].s);
	for(i=0; i<rxt; i++)//统计所有原45度的长度 
		fitness+=(r_x[i].e-r_x[i].s);
	for(i=0; i<ryt; i++)//统计所有原135度的长度 
		fitness+=(r_y[i].e-r_y[i].s);
	return fitness;
}
/*int main(){
	double point[][2]={0,0,0,0,1,2,2,1,5,0,2,-1};//point[0][0]为无效数据 
	int n = 5;
	int xes[]={0,1,2,0,3,2,0,4,3,1,3,5,1};//xes[0]为无效数据 
	cout<<Getfitness(point,xes,n)<<endl;
	return 0;
}*/
