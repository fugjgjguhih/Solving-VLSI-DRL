typedef struct HNode * heap;//�ṹ��ָ��
struct node{
	double xx;
	double yy;
};
struct HNode{
    node *Data;//��ʾ�ѵ����� ��СҪ���û������Ԫ�ظ�����+1
    int Size;//���������е�Ԫ��(������a[0]) 
    int Capacity; //�������������
}minHeap;
bool cmp1(struct node a,struct node b)
{
	if(a.xx<b.xx) return true;
	else return false;
}
bool cmp2(struct node a,struct node b)
{
	if(a.yy<b.yy) return true;
	else return false;
}
typedef heap MaxHeap; //����һ������
bool insertMinHeap(minHeap heap, int x){
    //�ж��Ƿ�����
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; heap->data[p/2]<x; p/=2) {
		//��������С��  ������a[0]λ�õĸ��ڱ����˱�����������Ԫ�ض�С��Ԫ��
        heap->data[p] = heap->data[p/2];
    }
    heap->data[p] = x;
    return true;
}
int deleteFromMinHeapy(minHeap heap){
    int top = heap->data[1];
    int last = heap->data[heap->Size--];
    int parent,child;
    for (parent = 1; parent*2<heap->Size; parent=child) {
        child = parent*2;
            //ע�������Ǵ������ӽڵ� ���� ���ӽڵ�����ӽڵ�С    
        if (child!=heap->Size && heap->data[child] > heap->data[child+1]) {
            child++;
            //��������ӽڵ㻹С
            if (heap->data[child]>last) {
                break;
            }else{//����
                heap->data[parent] = heap->data[child];
            }
        }
    }
    heap->data[parent] = last;
    return top;
}
bool insertMinHeapy(minHeap heap, node x){
    //�ж��Ƿ�����
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; heap->data[p/2]<x; p/=2) {
		//��������С��  ������a[0]λ�õĸ��ڱ����˱�����������Ԫ�ض�С��Ԫ��
        heap->data[p] = heap->data[p/2];
    }
    heap->data[p] = x;
    return true;
}

bool insertMinHeapx(minHeap heap, node x){
    //�ж��Ƿ�����
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; cmp1(heap->data[p/2],x); p/=2) {
		//��������С��  ������a[0]λ�õĸ��ڱ����˱�����������Ԫ�ض�С��Ԫ��
        heap->data[p] = heap->data[p/2];
    }
    heap->data[p] = x;
    return true;
}
int deleteFromMinHeapy(minHeap heap){
    int top = heap->data[1];
    int last = heap->data[heap->Size--];
    int parent,child;
    for (parent = 1; parent*2<heap->Size; parent=child) {
        child = parent*2;
            //ע�������Ǵ������ӽڵ� ���� ���ӽڵ�����ӽڵ�С    
        if (child!=heap->Size && heap->data[child] > heap->data[child+1]) {
            child++;
            //��������ӽڵ㻹С
            if (heap->data[child]>last) {
                break;
            }else{//����
                heap->data[parent] = heap->data[child];
            }
        }
    }
    heap->data[parent] = last;
    return top;
}
int deleteFromMinHeapx(minHeap heap){
    int top = heap->data[1];
    int last = heap->data[heap->Size--];
    int parent,child;
    for (parent = 1; parent*2<heap->Size; parent=child) {
        child = parent*2;
            //ע�������Ǵ������ӽڵ� ���� ���ӽڵ�����ӽڵ�С    
        if (child!=heap->Size && cmp1(heap->data[child+1], heap->data[child])) {
            child++;
            //��������ӽڵ㻹С
            if (cmp1(last,heap->data[child])) {
                break;
            }else{//����
                heap->data[parent] = heap->data[child];
            }
        }
    }
    heap->data[parent] = last;
    return top;
}
