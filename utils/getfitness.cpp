4/*#include<bits/stdc++.h>
#include<algorithm>
using namespace std; */
struct seg1
{
	double pos;
	double s;
	double e;
} xsegment[1010],ysegment[1010],r_x[1010],r_y[1010];
//ˮƽ��[y���� �����x ���յ�x],���߶�[x���� �����y ���յ�y]
struct seg2
{
	double sx;
	double sy;
	double ex;
	double ey;
} x45[1010],x135[1010];
//б�߶�[���x���꣬���y���꣬�յ�x���꣬�յ�y����]
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
	for(i=1; i<num; i++)//n�������n-1���Դ����� 
	{//xes_arry[i] ��������������Ϊһ���ʾxes_arry [������ �յ���� ���ӷ�ʽ] 
		//point_arry[�����p][0] p���x����;point_arry[�����p][1]p���y����
		x1=point_arry[xes_arry[i*3-2]][0]; //���x
		y1=point_arry[xes_arry[i*3-2]][1]; //���y
		x2=point_arry[xes_arry[i*3-1]][0]; //�յ�x
		y2=point_arry[xes_arry[i*3-1]][1]; //�յ�y
		sp=xes_arry[i*3]; //���ӷ�ʽ  sp=0��ֱ��/sp=1б�� 
		fx=fabs(x1-x2);  //ˮƽ����
		fy=fabs(y1-y2);  //��ֱ����
//		printf("%lf %lf %lf %lf %d %lf %lf\n",x1,y1,x2,y2,sp,fx,fy);
		
		if(sp==0)//���߶�����������ʱ ����ʱ����ĩ��Ϊ���ߣ� 
		{
			xsegment[xt].pos=y1;
			ysegment[yt].pos=x2;
			if(x2<x1)
			{
				xsegment[xt].s=x2; //ˮƽ������������
				xsegment[xt++].e=x1; //ˮƽ�����յ������
			}
			else
			{
				xsegment[xt].s=x1;
				xsegment[xt++].e=x2;
			}
			if(y2<y1)
			{
				ysegment[yt].s=y2; //��ֱ�������������
				ysegment[yt++].e=y1; //��ֱ�����յ�������
			}
			else
			{
				ysegment[yt].s=y1;
				ysegment[yt++].e=y2;
			}
			
		}
		if(sp==1) //���߶ε����ӷ�ʽΪ�������ٽṹʱ
		{
			if(fx==0||fy==0) //���㹲�ߵ����
			{
				if(fy==0) //ˮƽ���� 
				{
					xsegment[xt].pos=y1;
					if(x1>x2)
					{
						xsegment[xt].s=x2; //ˮƽ������������
						xsegment[xt++].e=x1; //ˮƽ�����յ������
					}
					else
					{
						xsegment[xt].s=x1;
						xsegment[xt++].e=x2;
					}
				}
				else//��ֱ���� 
				{
					ysegment[yt].pos=x1;
					if(y1>y2)
					{
						ysegment[yt].s=y2; //��ֱ�������������
						ysegment[yt++].e=y1; //��ֱ�����յ�������
					}
					else
					{
						ysegment[yt].s=y1;
						ysegment[yt++].e=y2;
					}
				}
			}
			else //���㲻����
			{
				if(fx>fy) //ˮƽ�����ڴ�ֱ�������
				{
					if(x1<x2)
					{
						xsegment[xt].pos=y1;
						xsegment[xt].s=x1;
						xsegment[xt++].e=x2-fy;
						if(y1<y2) //��¼45���߶ε������յ�����
						{
							x45[x45t].sx=x2-fy;
							x45[x45t].sy=y1;
							x45[x45t].ex=x2;
							x45[x45t++].ey=y2;
						}
						else  //��¼135���߶ε������յ�����
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
						if(y1>y2) //��¼45���߶ε������յ�����
						{
							x45[x45t].sx=x2;
							x45[x45t].sy=y2;
							x45[x45t].ex=x2+fy;
							x45[x45t++].ey=y1;
						}
						else  //��¼135���߶ε������յ�����
						{
							x135[x135t].sx=x2+fy;
							x135[x135t].sy=y1;
							x135[x135t].ex=x2;
							x135[x135t++].ey=y2;
						}
					} 
				} 
				else //��ֱ�����ڻ����ˮƽ�������
				{
					if(y1>y2)
					{
						ysegment[yt].pos=x1;
						ysegment[yt].s=y2+fx;
						ysegment[yt++].e=y1;
						if(x2<x1) //��¼45���߶ε������յ�����
						{
							x45[x45t].sx=x2;
							x45[x45t].sy=y2;
							x45[x45t].ex=x1;
							x45[x45t++].ey=y2+fy;
						}
						else  //��¼135���߶ε������յ�����
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
						if(x2>x1) //��¼45���߶ε������յ�����
						{
							x45[x45t].sx=x1;
							x45[x45t].sy=y2-fy;
							x45[x45t].ex=x2;
							x45[x45t++].ey=y2;
						}
						else  //��¼135���߶ε������յ�����
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
	for(i=0; i<x45t; i++) //��45����˳ʱ����ת45�ȱ�Ϊˮƽ��������xsegment��
	{
		//��45�����ӳ�����y�ύ��Ϊԭ�㣨Բ�ģ���ת  p=0.5*sqrt��2�� 
		r_x[rxt].pos=P*(x45[i].sy-x45[i].sx); //��ת���y����
		r_x[rxt].s=P*(x45[i].sx+x45[i].sy);   //��ת������x���� 
		r_x[rxt++].e=P*(x45[i].ex+x45[i].ey); //��ת����յ�x����
	}
		for(i=0; i<x135t; i++) //��135����˳ʱ����ת45�ȱ�Ϊ��ֱ��������ysegment��
	{
		//���д����ӳ������ύ��Ϊԭ�㣨Բ�ģ���ת  p=0.5*sqrt��2�� 
		r_y[ryt].pos=P*(x135[i].sx+x135[i].sy);
		r_y[ryt].s=P*(x135[i].sy-x135[i].sx);
		r_y[ryt++].e=P*(x135[i].ey-x135[i].ex);
	}
	/*������������£���ʡ��*/ 
	for(i=0; i<xt; i++) //����ˮƽ�߶Σ�ʹ����������յ�����
	{
		if(xsegment[i].s>xsegment[i].e)
		{
			midx=xsegment[i].s;
			xsegment[i].s=xsegment[i].e;
			xsegment[i].e=midx;
		}
	}
	for(i=0; i<yt; i++)//������ֱ�߶Σ�ʹ��������£��յ�����
	{
		if(ysegment[i].s>ysegment[i].e)
		{
			midy=ysegment[i].s;
			ysegment[i].s=ysegment[i].e;
			ysegment[i].e=midy;
		}
	}
	for(i=0; i<rxt; i++)//����ˮƽ�߶Σ�ʹ����������յ�����
	{
		if(r_x[i].s>r_x[i].e)
		{
			midx=r_x[i].s;
			r_x[i].s=r_x[i].e;
			r_x[i].e=midx;
		}
	}
	for(i=0; i<ryt; i++)//������ֱ�߶Σ�ʹ��������£��յ�����
	{
		if(r_y[i].s>r_y[i].e)
		{
			midy=r_y[i].s;
			r_y[i].s=r_y[i].e;
			r_y[i].e=midy;
		}
	}
	/*��pos��s����洢*/
	sort(xsegment,xsegment+xt,cmp1);  //���յ�һ��pos����������
	sort(ysegment,ysegment+yt,cmp1);
	sort(r_x,r_x+rxt,cmp1);
	sort(r_y,r_y+ryt,cmp1);
	for(i=1; i<xt; i++) //ˮƽ����pos����Ļ����ϣ��ٰ���s��������
	{
		k=i-1;
		while(i<xt&&xsegment[i].pos==xsegment[i-1].pos)
			i++;
		sort(xsegment+k,xsegment+i,cmp2);
	}
	for(i=1; i<yt; i++) //��ֱ����pos����Ļ����ϣ��ٰ���s��������
	{
		k=i-1;
		while(i<yt&&ysegment[i].pos==ysegment[i-1].pos)
			i++;
		sort(ysegment+k,ysegment+i,cmp2);
	}
	for(i=1; i<rxt; i++) //��ת���ˮƽ����pos����Ļ����ϣ��ٰ���s��������
	{
		k=i-1;
		while(i<rxt&&r_x[i].pos==r_x[i-1].pos)
			i++;
		sort(r_x+k,r_x+i,cmp2);
	}
	for(i=1; i<ryt; i++) //��ת��Ĵ�ֱ����pos����Ļ����ϣ��ٰ���s��������
	{
		k=i-1;
		while(i<ryt&&r_y[i].pos==r_y[i-1].pos)
			i++;
		sort(r_y+k,r_y+i,cmp2);
	}
	/*�ϲ��ص�����*/ 
	for(i=1; i<xt; i++) //�ϲ��ص���ԭˮƽ��
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
	for(i=1; i<yt; i++) //�ϲ��ص���ԭ��ֱ��
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
	for(i=1; i<rxt; i++) //�ϲ���ת���ص���ˮƽ��
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
	for(i=1; i<ryt; i++) //�ϲ���ת���ص��Ĵ�ֱ��
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
	
	for(i=0; i<xt; i++)//ͳ������ԭˮƽ�ߵĳ��� 
		fitness+=(xsegment[i].e-xsegment[i].s);
	for(i=0; i<yt; i++)//ͳ������ԭ��ֱ�ߵĳ��� 
		fitness+=(ysegment[i].e-ysegment[i].s);
	for(i=0; i<rxt; i++)//ͳ������ԭ45�ȵĳ��� 
		fitness+=(r_x[i].e-r_x[i].s);
	for(i=0; i<ryt; i++)//ͳ������ԭ135�ȵĳ��� 
		fitness+=(r_y[i].e-r_y[i].s);
	return fitness;
}
/*int main(){
	double point[][2]={0,0,0,0,1,2,2,1,5,0,2,-1};//point[0][0]Ϊ��Ч���� 
	int n = 5;
	int xes[]={0,1,2,0,3,2,0,4,3,1,3,5,1};//xes[0]Ϊ��Ч���� 
	cout<<Getfitness(point,xes,n)<<endl;
	return 0;
}*/
