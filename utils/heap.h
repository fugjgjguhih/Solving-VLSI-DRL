typedef struct HNode * heap;//结构体指针
struct node{
	double xx;
	double yy;
};
struct HNode{
    node *Data;//表示堆的数组 大小要在用户输入的元素个数上+1
    int Size;//数组里已有的元素(不包含a[0]) 
    int Capacity; //数组的数量上限
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
typedef heap MaxHeap; //定义一个最大堆
bool insertMinHeap(minHeap heap, int x){
    //判断是否满了
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; heap->data[p/2]<x; p/=2) {
		//这里是最小堆  所以在a[0]位置的岗哨保存了比数组中所有元素都小的元素
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
            //注意这里是存在右子节点 并且 右子节点比左子节点小    
        if (child!=heap->Size && heap->data[child] > heap->data[child+1]) {
            child++;
            //如果比右子节点还小
            if (heap->data[child]>last) {
                break;
            }else{//下滤
                heap->data[parent] = heap->data[child];
            }
        }
    }
    heap->data[parent] = last;
    return top;
}
bool insertMinHeapy(minHeap heap, node x){
    //判断是否满了
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; heap->data[p/2]<x; p/=2) {
		//这里是最小堆  所以在a[0]位置的岗哨保存了比数组中所有元素都小的元素
        heap->data[p] = heap->data[p/2];
    }
    heap->data[p] = x;
    return true;
}

bool insertMinHeapx(minHeap heap, node x){
    //判断是否满了
    if (heap->Size == heap->Capacity){
        return false;
    }
    int p = ++heap->Size;
    for (; cmp1(heap->data[p/2],x); p/=2) {
		//这里是最小堆  所以在a[0]位置的岗哨保存了比数组中所有元素都小的元素
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
            //注意这里是存在右子节点 并且 右子节点比左子节点小    
        if (child!=heap->Size && heap->data[child] > heap->data[child+1]) {
            child++;
            //如果比右子节点还小
            if (heap->data[child]>last) {
                break;
            }else{//下滤
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
            //注意这里是存在右子节点 并且 右子节点比左子节点小    
        if (child!=heap->Size && cmp1(heap->data[child+1], heap->data[child])) {
            child++;
            //如果比右子节点还小
            if (cmp1(last,heap->data[child])) {
                break;
            }else{//下滤
                heap->data[parent] = heap->data[child];
            }
        }
    }
    heap->data[parent] = last;
    return top;
}
