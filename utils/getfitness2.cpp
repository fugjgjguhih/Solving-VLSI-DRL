#include "uthash.h"
#include "heap.h"
typedef struct {
    int key;
    minheap value;
    UT_hash_handle hh;
} HashNode;
typedef HashNode* HashHead;
HashNode *find_node(HashHead head, int node_id) {
    HashNode *s;
    HASH_FIND_INT(head, &node_id, s);  /* s: output pointer */
    return s;
}	
void add_node(int mykey, char *value) {  
    struct HashNode *s;  
  
    HASH_FIND_INT(g_nodes, &mykey, s);  /* mykey already in the hash? */  
    if (s==NULL) {  
      s = (struct HashNode*)malloc(sizeof(struct HashNode));  
      s->ikey = mykey;  
      HASH_ADD_INT(g_nodes, ikey, s);  /* ikey: name of key field */  
    }  
    strcpy(s->value, value);  
}
void delete_node(HashHead *head,HashNode *node) {
    if (node) {
        HASH_DEL(*head, node);  /* node: pointer to deletee */
        free(node);             /* optional; it's up to you! */
    }
}
double eval(double* points, int* xes , int num){
	double len = 0 ;
	double x[num]; double y[num];
	int i ,j;
	HashNode* xx=NULL;
	HashNode* yy=NULL;
	HashNode* ne=NULL;
	HashNode* nw=NULL;
	for(i = 0,j =0 ; i<num;){
		x[i] = points[j++];
        y[i] = points[j++];
        i++;
	}
	for(i=1; i<num; i++){
		s = xes[i*3-1];
		sp = xes[i*3-2]; dp = xes[i*3-1];
		sx = x[sp], dx = x[dp];
		sy = y[sp], sp = y[dp];
		HashNode* xp = find_node(HashNode xx,sy);
		if(s==NULL){
			 add_node();
		}
	 }
}
