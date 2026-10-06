#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

typedef long long int llint;
char *instance_name;
llint source_set_size = 1;
char *instance_path;
/* To compile to use with C-types:
gcc -fPIC -shared -o bareHPF.so bareHPF.c -- Linux
gcc -shared -Wl,-install_name bareHPF.so -o bareHPF.so -fPIC bareHPF.c --MACOS
*/

float *_params = NULL;
float TOL = 1e-6;
float
timer (void)
{
  struct rusage r;

  getrusage(0, &r);
  return (float) (r.ru_utime.tv_sec + r.ru_utime.tv_usec / (float)1000000);
}

struct node;

typedef struct arc
{
  float flow;
  float capacity;
	float wt;
	float cst;
	int from;
	int to;
	char direction;
 // float *capacities;
} Arc;

typedef struct node
{
	Arc **outOfTree;
	Arc *arcToParent;
	struct node *parent;
	struct node *childList;
	struct node *nextScan;
	struct node *next;
	struct node *prev;
  float excess;
	int visited;
	int numAdjacent;
	//int number;
	int label;
	int numOutOfTree;
	int nextArc;
	int breakpoint;
} Node;


typedef struct root
{
	Node *start;
	Node *end;
} Root;

//---------------  Global variables ------------------
static int numNodes = 0;
static int numArcs = 0;
static int source = 0;
static int sink = 1;
static int numParams = 0;
static float minParam = 0;
static float maxParam = 0;

static int highestStrongLabel = 1;

static Node *adjacencyList = NULL;
static Root *strongRoots = NULL;
static int *labelCount = NULL;
static Arc *arcList = NULL;
//-----------------------------------------------------

#ifdef STATS
static llint numPushes = 0;
static int numMergers = 0;
static int numRelabels = 0;
static int numGaps = 0;
static llint numArcScans = 0;
#endif

//#define USE_ARC_MACRO
#ifdef USE_ARC_MACRO

#define numax(a,b) (a>b ? a : b)

#define zero_floor(a) ( numax(a,0) )

// TODO: change conditions, they are too expensive
//#define computeArcCapacity(a,p) (a->from==(source) ?      \
																						zero_floor(a->wt-p*(a->cst)) :  \
																				(a->to==(sink) ? \
																					  zero_floor(p*(a->cst)-a->wt) :  \
																						a->capacity) )

#define computeArcCapacity(a,p) ((a->wt)+zero_floor(p*(a->cst)))

#else

static float
numax (float a, float b)
{
	return a>b ? a : b;
}

static float
computeArcCapacity(Arc *ac, float param)
{
  return ac->wt + numax(  param*(ac->cst), 0);
}


#endif

static void
initializeNode (Node *nd, const int n)
{
	nd->label = 0;
	nd->excess = 0;
	nd->parent = NULL;
	nd->childList = NULL;
	nd->nextScan = NULL;
	nd->nextArc = 0;
	nd->numOutOfTree = 0;
	nd->arcToParent = NULL;
	nd->next = NULL;
	nd->prev = NULL;
	nd->visited = 0;
	nd->numAdjacent = 0;
	//nd ->number = n;
	nd->outOfTree = NULL;
	nd->breakpoint = (numParams+1);
}

static void
initializeRoot (Root *rt)
{
	rt->start = (Node *) malloc (sizeof(Node));
	rt->end = (Node *) malloc (sizeof(Node));

	if ((rt->start == NULL) || (rt->end == NULL))
	{
		printf ("%s Line %d: Out of memory\n", __FILE__, __LINE__);
		exit (1);
	}

	initializeNode (rt->start, 0);
	initializeNode (rt->end, 0);

	rt->start->next = rt->end;
	rt->end->prev = rt->start;
}


static void
freeRoot (Root *rt)
{
	free(rt->start);
	rt->start = NULL;

	free(rt->end);
	rt->end = NULL;
}

static void
liftAll (Node *rootNode, const int theparam)
{
	Node *temp, *current=rootNode;

	current->nextScan = current->childList;

	-- labelCount[current->label];
	current->label = numNodes;
	current->breakpoint = (theparam+1);

	source_set_size++;
	for ( ; (current); current = current->parent)
	{
		while (current->nextScan)
		{
			temp = current->nextScan;
			current->nextScan = current->nextScan->next;
			current = temp;
			current->nextScan = current->childList;

			-- labelCount[current->label];
			current->label = numNodes;
			current->breakpoint = (theparam+1);
			source_set_size++;
		}
	}
}

static void
addToStrongBucket (Node *newRoot, Node *rootEnd)
{
	newRoot->next = rootEnd;
	newRoot->prev = rootEnd->prev;
	rootEnd->prev = newRoot;
	newRoot->prev->next = newRoot;
}

static void
createOutOfTree (Node *nd)
{
	if (nd->numAdjacent)
	{
		if ((nd->outOfTree = (Arc **) malloc (nd->numAdjacent * sizeof (Arc *))) == NULL)
		{
			printf ("%s Line %d: Out of memory\n", __FILE__, __LINE__);
			exit (1);
		}
	}
}

static void
initializeArc (Arc *ac)
{
	ac->from = 0;
	ac->to = 0;
	ac->capacity = 0;
	ac->flow = 0;
	ac->direction = 1;
	//ac->capacities = NULL;
}

static void
addOutOfTreeNode (Node *n, Arc *out)
{
	n->outOfTree[n->numOutOfTree] = out;
	++ n->numOutOfTree;
}

static void readData(int origNumNodes, int origNumArcs, int numLambda,
	float * eds, int sourceID, int sinkID, float * params)
/*************************************************************************
readData
*************************************************************************/
{
	_params = params;
  int i, j, from, to, first=0;
  float capacity, cst, wt, step;
	char ch, ch1;
	char *word, *line, *tmpline;
	int a_i, b_i;
	source = sourceID;
	sink = sinkID;
	Arc *ac = NULL;
  numArcs = origNumArcs;
	numNodes = origNumNodes;
  numParams = numLambda;
	//printf("c No nodes is %d, No arcs is %d, No parameters is %d\n",
    //      numNodes, numArcs, numParams);
	if ((adjacencyList = (Node *) malloc (numNodes * sizeof (Node))) == NULL)
	{
		printf ("%s, %d: Could not allocate memory.\n", __FILE__, __LINE__);
		exit (1);
	}

	if ((strongRoots = (Root *) malloc (numNodes * sizeof (Root))) == NULL)
	{
		printf ("%s, %d: Could not allocate memory.\n", __FILE__, __LINE__);
		exit (1);
	}

	if ((labelCount = (int *) malloc (numNodes * sizeof (int))) == NULL)
	{
		printf ("%s, %d: Could not allocate memory.\n", __FILE__, __LINE__);
		exit (1);
	}

	if ((arcList = (Arc *) malloc (numArcs * sizeof (Arc))) == NULL)
	{
		printf ("%s, %d: Could not allocate memory.\n", __FILE__, __LINE__);
		exit (1);
	}

	for (i=0; i<numNodes; ++i)
	{
		initializeRoot (&strongRoots[i]);
		labelCount[i] = 0;
	}
	// Initialize root and sink with weight 0 and cost 0
	for (i=0; i<numNodes; i++)
	{
		initializeNode (&adjacencyList[i], i);
	}

	for (i=0; i<numArcs; ++i)
	{
		initializeArc (&arcList[i]);
	}
	//printf("c Finished reading the first line\n");
	// printf("c origNumArcs = %d and origNumNodes = %d\n", origNumArcs, origNumNodes);
	// INITIALIZE THE ARCS
  for (i= 0; i < origNumArcs; ++ i)
  {
  	// printf("c I am HERE\n");
  	ac = &arcList[first];
  	//printf("from %lf to %lf capacity %lf\n", eds[i*3], eds[i*3+1], eds[i*3+2]);
  	from = (int) eds[i*4];
  	to = (int) eds[i*4+1];
  	ac->wt = eds[i*4+2];
  	ac->cst = eds[i*4+3];
  	// printf("a %d %d %lf\n", from, to , capacity);
  	ac->from = from;
  	// adjacencyList[from+2].wt += capacity;
  	ac->to = to;
  	// adjacencyList[to+2].wt += capacity;
		/*
  	if ((ac->capacities = (float *) malloc (sizeof (float))) == NULL)
  	{
  		printf ("%s Line %d: Out of memory\n", __FILE__, __LINE__);
  		exit (1);
  	}
  	ac->capacities[0] = capacity;
		*/
		ac->capacity = ac->wt + numax(0, params[0]*ac->cst);
  	++ adjacencyList[ac->from].numAdjacent;
  	++ adjacencyList[ac->to].numAdjacent;
  	++first;
  }
	//printf("c Finished creating arcs\n");

	// printf("Finished with my changes and first equals %d\n", first);

	for (i=0; i<numNodes; ++i)
	{
		createOutOfTree (&adjacencyList[i]);
	}
	// printf("Finished create outtrees\n");
	for (i=0; i<numArcs; i++)
	{
		to = arcList[i].to ;
		from = arcList[i].from ;
		capacity = arcList[i].capacity;

		if (!((source == to) || (sink == from) || (from == to)))
		{
			if ((source == from) && (to == sink))
			{
				arcList[i].flow = capacity;
			}
			else if (from == source)
			{
				addOutOfTreeNode (&adjacencyList[from], &arcList[i]);
			}
			else if (to == sink)
			{
				addOutOfTreeNode (&adjacencyList[to], &arcList[i]);
			}
			else
			{
				addOutOfTreeNode (&adjacencyList[from], &arcList[i]);
			}
		}
	}
}

static void
simpleInitialization (void)
{
	int i, size;
	Arc *tempArc;

    size = adjacencyList[source].numOutOfTree;
    for (i=0; i<size; ++i)
	{
		tempArc = adjacencyList[source].outOfTree[i];
		tempArc->flow = tempArc->capacity;
		adjacencyList[tempArc->to].excess += tempArc->capacity;
	}

	size = adjacencyList[sink].numOutOfTree;
    for (i=0; i<size; ++i)
	{
		tempArc = adjacencyList[sink].outOfTree[i];
		tempArc->flow = tempArc->capacity;
		adjacencyList[tempArc->from].excess -= tempArc->capacity;
	}

	adjacencyList[source].excess = 0;
	adjacencyList[sink].excess = 0;

	for (i=0; i<numNodes; ++i)
	{
		if (adjacencyList[i].excess > 0)
		{
		    adjacencyList[i].label = 1;
			++ labelCount[1];

			addToStrongBucket (&adjacencyList[i], strongRoots[1].end);
		}
	}

	adjacencyList[source].label = numNodes;
	adjacencyList[source].breakpoint = 0;
	adjacencyList[sink].label = 0;
	adjacencyList[sink].breakpoint = (numParams+2);
	labelCount[0] = (numNodes - 2) - labelCount[1];
}

static inline int 
addRelationship (Node *newParent, Node *child) 
{
  child->parent = newParent;
  child->next = newParent->childList;
	
	if (newParent->childList != NULL)
	{
		newParent->childList->prev = child;
	}

  newParent->childList = child;

  return 0;
}

static inline void
breakRelationship (Node *oldParent, Node *child) 
{
  Node *current;

  child->parent = NULL;

  if (oldParent->childList == child) 
  {
    oldParent->childList = child->next;
    child->next = NULL;
    return;
  }

	current = child->prev;

  current->next = child->next;
	if(child->next != NULL) child->next->prev = current;
  child->next = NULL;
}


static void
merge (Node *parent, Node *child, Arc *newArc)
{
	Arc *oldArc;
	Node *current = child, *oldParent, *newParent = parent;

#ifdef STATS
	++ numMergers;
#endif

	while (current->parent)
	{
		oldArc = current->arcToParent;
		current->arcToParent = newArc;
		oldParent = current->parent;
		breakRelationship (oldParent, current);
		addRelationship (newParent, current);
		newParent = current;
		current = oldParent;
		newArc = oldArc;
		newArc->direction = 1 - newArc->direction;
	}

	current->arcToParent = newArc;
	addRelationship (newParent, current);
}


static inline void
//pushUpward (Arc *currentArc, Node *child, Node *parent, const int resCap)
pushUpward (Arc *currentArc, Node *child, Node *parent, const float resCap)
{
#ifdef STATS
	++ numPushes;
#endif

	if (resCap >= child->excess)
	{
		parent->excess += child->excess;
		currentArc->flow += child->excess;
		child->excess = 0;
		return;
	}

	currentArc->direction = 0;
	parent->excess += resCap;
	child->excess -= resCap;
	currentArc->flow = currentArc->capacity;
	parent->outOfTree[parent->numOutOfTree] = currentArc;
	++ parent->numOutOfTree;
	breakRelationship (parent, child);

	addToStrongBucket (child, strongRoots[child->label].end);
}


static inline void
//pushDownward (Arc *currentArc, Node *child, Node *parent, int flow)
pushDownward (Arc *currentArc, Node *child, Node *parent, float flow)
{
#ifdef STATS
	++ numPushes;
#endif

	if (flow >= child->excess)
	{
		parent->excess += child->excess;
		currentArc->flow -= child->excess;
		child->excess = 0;
		return;
	}

	currentArc->direction = 1;
	child->excess -= flow;
	parent->excess += flow;
	currentArc->flow = 0;
	parent->outOfTree[parent->numOutOfTree] = currentArc;
	++ parent->numOutOfTree;
	breakRelationship (parent, child);

	addToStrongBucket (child, strongRoots[child->label].end);
}

static void
pushExcess (Node *strongRoot)
{
	Node *current, *parent;
	Arc *arcToParent;

	for (current = strongRoot; (current->excess && current->parent); current = parent)
	{
		parent = current->parent;
		arcToParent = current->arcToParent;
		if (arcToParent->direction)
		{
			pushUpward (arcToParent, current, parent, (arcToParent->capacity - arcToParent->flow));
		}
		else
		{
			pushDownward (arcToParent, current, parent, arcToParent->flow);
		}
	}

	if (current->excess > 0)
	{
		if (!current->next)
		{
			addToStrongBucket (current, strongRoots[current->label].end);
		}
	}
}


static Arc *
findWeakNode (Node *strongNode, Node **weakNode)
{
	int i, size;
	Arc *out;

	size = strongNode->numOutOfTree;

	for (i=strongNode->nextArc; i<size; ++i)
	{

#ifdef STATS
		++ numArcScans;
#endif

		if (adjacencyList[strongNode->outOfTree[i]->to].label == (highestStrongLabel-1))
		{
			strongNode->nextArc = i;
			out = strongNode->outOfTree[i];
			(*weakNode) = &adjacencyList[out->to];
			-- strongNode->numOutOfTree;
			strongNode->outOfTree[i] = strongNode->outOfTree[strongNode->numOutOfTree];
			return (out);
		}
		else if (adjacencyList[strongNode->outOfTree[i]->from].label == (highestStrongLabel-1))
		{
			strongNode->nextArc = i;
			out = strongNode->outOfTree[i];
			(*weakNode) = &adjacencyList[out->from];
			-- strongNode->numOutOfTree;
			strongNode->outOfTree[i] = strongNode->outOfTree[strongNode->numOutOfTree];
			return (out);
		}
	}

	strongNode->nextArc = strongNode->numOutOfTree;

	return NULL;
}


static void
checkChildren (Node *curNode)
{
	for ( ; (curNode->nextScan); curNode->nextScan = curNode->nextScan->next)
	{
		if (curNode->nextScan->label == curNode->label)
		{
			return;
		}

	}

	-- labelCount[curNode->label];
	++	curNode->label;
	++ labelCount[curNode->label];

#ifdef STATS
	++ numRelabels;
#endif

	curNode->nextArc = 0;
}

static void
processRoot (Node *strongRoot)
{
	Node *temp, *strongNode = strongRoot, *weakNode;
	Arc *out;

	strongRoot->nextScan = strongRoot->childList;

	if ((out = findWeakNode (strongRoot, &weakNode)))
	{
		merge (weakNode, strongNode, out);
		pushExcess (strongRoot);
		return;
	}

	checkChildren (strongRoot);

	while (strongNode)
	{
		while (strongNode->nextScan)
		{
			temp = strongNode->nextScan;
			strongNode->nextScan = strongNode->nextScan->next;
			strongNode = temp;
			strongNode->nextScan = strongNode->childList;

			if ((out = findWeakNode (strongNode, &weakNode)))
			{
				merge (weakNode, strongNode, out);
				pushExcess (strongRoot);
				return;
			}

			checkChildren (strongNode);
		}

		if ((strongNode = strongNode->parent))
		{
			checkChildren (strongNode);
		}
	}

	addToStrongBucket (strongRoot, strongRoots[strongRoot->label].end);

	++ highestStrongLabel;
}

static Node *
getHighestStrongRoot (const int theparam)
{
	int i;
	Node *strongRoot;

	for (i=highestStrongLabel; i>0; --i)
	{
		if (strongRoots[i].start->next != strongRoots[i].end)
		{
			highestStrongLabel = i;
			if (labelCount[i-1])
			{
				strongRoot = strongRoots[i].start->next;
				strongRoot->next->prev = strongRoot->prev;
				strongRoot->prev->next = strongRoot->next;
				strongRoot->next = NULL;
				return strongRoot;
			}

			while (strongRoots[i].start->next != strongRoots[i].end)
			{

#ifdef STATS
				++ numGaps;
#endif
				strongRoot = strongRoots[i].start->next;
				strongRoot->next->prev = strongRoot->prev;
				strongRoot->prev->next = strongRoot->next;
				liftAll (strongRoot, theparam);
			}
		}
	}

	if (strongRoots[0].start->next == strongRoots[0].end)
	{
		return NULL;
	}

	while (strongRoots[0].start->next != strongRoots[0].end)
	{
		strongRoot = strongRoots[0].start->next;
		strongRoot->next->prev = strongRoot->prev;
		strongRoot->prev->next = strongRoot->next;

		strongRoot->label = 1;
		-- labelCount[0];
		++ labelCount[1];

#ifdef STATS
		++ numRelabels;
#endif

		addToStrongBucket (strongRoot, strongRoots[strongRoot->label].end);
	}

	highestStrongLabel = 1;

	strongRoot = strongRoots[1].start->next;
	strongRoot->next->prev = strongRoot->prev;
	strongRoot->prev->next = strongRoot->next;
	strongRoot->next = NULL;

	return strongRoot;
}

static void
updateCapacities (const int theparam)
{
	int i, size;
	//int delta;
    float delta;
	Arc *tempArc;
	//Node *tempNode;

	float param = _params[theparam];//lambdaStart + ((lambdaEnd-lambdaStart) *  ((float)theparam/(numParams-1)));
	size = adjacencyList[source].numOutOfTree;
	for (i=0; i<size; ++i)
	{
		tempArc = adjacencyList[source].outOfTree[i];
		//delta = (tempArc->capacities[theparam] - tempArc->capacity);
		delta = ( computeArcCapacity(tempArc, param) - tempArc->capacity);
		if (delta < 0)
		{
      printf ("c Error on source-adjacent arc (%d, %d): capacity decreases by %lf (%lf minus %lf) at parameter %d.\n",
          tempArc->from ,
          tempArc->to ,
          (-delta),
          computeArcCapacity(tempArc, param), //tempArc->capacities[theparam],
          tempArc->capacity,
          (theparam+1));
			exit(0);
		}

		tempArc->capacity += delta;
		tempArc->flow += delta;
		adjacencyList[tempArc->to].excess += delta;

		if ((adjacencyList[tempArc->to].label < numNodes) && (adjacencyList[tempArc->to].excess > 0))
		{
			pushExcess (&adjacencyList[tempArc->to]);
		}
	}

	size = adjacencyList[sink].numOutOfTree;
	for (i=0; i<size; ++i)
	{
		tempArc = adjacencyList[sink].outOfTree[i];
		//delta = (tempArc->capacities[theparam] - tempArc->capacity);
		delta = (computeArcCapacity(tempArc, param) - tempArc->capacity);
		if (delta > 0)
		{
			/*printf ("c Error on sink-adjacent arc (%d, %d): capacity %d increases to %d at parameter %d.\n",
				tempArc->from ,
				tempArc->to ,
				tempArc->capacity,
				tempArc->capacities[theparam],
				(theparam+1));*/
            /*printf ("c Error on sink-adjacent arc (%d, %d): capacity %lf increases to %lf at parameter %d.\n",
                tempArc->from,
                tempArc->to,
                tempArc->capacity,
          			computeArcCapacity(tempArc, param), //tempArc->capacities[theparam],
                (theparam+1));
			exit(0);*/
		}

		tempArc->capacity += delta;
		tempArc->flow += delta;
		adjacencyList[tempArc->from].excess -= delta;

		if ((adjacencyList[tempArc->from].label < numNodes) && (adjacencyList[tempArc->from].excess > 0))
		{
			pushExcess (&adjacencyList[tempArc->from]);
		}
	}

	highestStrongLabel = (numNodes-1);
}

static float
computeMinCut (void)
{
    //int i, mincut=0;
    int i;
    float mincut=0;

	for (i=0; i<numArcs; ++i)
	{
		if ((adjacencyList[arcList[i].from].label >= numNodes) && (adjacencyList[arcList[i].to].label < numNodes))
		{
			mincut += arcList[i].capacity;
		}
	}
	return mincut;
}

static void
pseudoflowPhase1 (void)
{
	Node *strongRoot;
	int theparam = 0;
	float thetime;

	thetime = timer ();
	while ((strongRoot = getHighestStrongRoot (theparam)))
	{
		processRoot (strongRoot);
	}
	/*printf ("c Finished solving parameter %d\nc Flow: %lf\nc Elapsed time: %.3lf\n",
		(theparam+1),
		computeMinCut (),
		(timer () - thetime));*/

	llint orig_ss = 0;
	if(orig_ss!=source_set_size)
		//printf("c for lambda=%lf #source_set=%lld\n", _params[theparam], source_set_size);
	orig_ss = source_set_size ;
	for (theparam=1; theparam < numParams; ++ theparam)
	{
		updateCapacities (theparam);
#ifdef PROGRESS
		printf ("c Finished updating capacities and excesses.\n");
		fflush (stdout);
#endif
		while ((strongRoot = getHighestStrongRoot (theparam)))
		{
			processRoot (strongRoot);
		}

	  if(orig_ss!=source_set_size)
		  //printf("c for lambda=%lf #source_set=%lld\n", _params[theparam], source_set_size);
	  orig_ss = source_set_size ;
	  //printf("c for lambda=%lf #source_set=%lfld", theparam, source_set_size);
		// if (theparam < 4)
		// {
		// 	printf ("c Finished parameter: %d\nc Flow: %lf\nc Elapsed time: %.3lf\n",
		// 	 (theparam+1),
		// 		computeMinCut (),
		// 		(timer () - thetime));
		// }
	}
}

static void
checkOptimality (void)
{
	int i, check = 1;
	//llint mincut = 0, *excess;
    float mincut=0, *excess;

	//excess = (llint *) malloc (numNodes * sizeof (llint));
    excess = (float *) malloc (numNodes * sizeof (float));
	if (!excess)
	{
		printf ("%s Line %d: Out of memory\n", __FILE__, __LINE__);
		exit (1);
	}

	for (i=0; i<numNodes; ++i)
	{
		excess[i] = 0;
	}

	for (i=0; i<numArcs; ++i)
	{
		if ((adjacencyList[arcList[i].from].label >= numNodes) && (adjacencyList[arcList[i].to].label < numNodes))
		{
			mincut += arcList[i].capacity;
		}

		if ((arcList[i].flow > arcList[i].capacity) || (arcList[i].flow < 0))
		{
			check = 0;
			printf("c Capacity constraint violated on arc (%d, %d)\n",
				arcList[i].from ,
				arcList[i].to );
		}
		excess[arcList[i].from] -= arcList[i].flow;
		excess[arcList[i].to] += arcList[i].flow;
	}

	for (i=0; i<numNodes; i++)
	{
		if ((i != (source)) && (i != (sink)))
		{
			if (excess[i])
			{
				check = 0;
				/*printf ("c Flow balance constraint violated in node %d. Excess = %lld\n",
					i+1,
					excess[i]);*/
                //printf("c Flow balance constraint violated in node %d, Excess = %lf\n", i+1, excess[i]);
			}
		}
	}

	if (check)
	{
		printf ("c\nc Solution checks as feasible.\n");
	}

	check = 1;

	if (excess[sink] != mincut)
	{
		check = 0;
		printf("c Flow is not optimal - max flow does not equal min cut!\nc\n");
	}

	if (check)
    {
        printf ("c\nc Solution checks as optimal.\nc \n");
        //printf ("s Max Flow            : %lld\n", mincut);
        printf ("s Max Flow            : %lf\n", mincut);
    }
	free (excess);
	excess = NULL;
}


static void
quickSort (Arc **arr, const int first, const int last)
{
	int i, j, left=first, right=last, x1, x2, x3, mid, pivot, pivotval;
	Arc *swap;

	if ((right-left) <= 5)
	{// Bubble sort if 5 elements or less
		for (i=right; (i>left); --i)
		{
			swap = NULL;
			for (j=left; j<i; ++j)
			{
				if (arr[j]->flow < arr[j+1]->flow)
				{
					swap = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = swap;
				}
			}

			if (!swap)
			{
				return;
			}
		}

		return;
	}

	mid = (first+last)/2;

	x1 = arr[first]->flow;
	x2 = arr[mid]->flow;
	x3 = arr[last]->flow;

	pivot = mid;

	if (x1 <= x2)
	{
		if (x2 > x3)
		{
			pivot = left;

			if (x1 <= x3)
			{
				pivot = right;
			}
		}
	}
	else
	{
		if (x2 <= x3)
		{
			pivot = right;

			if (x1 <= x3)
			{
				pivot = left;
			}
		}
	}

	pivotval = arr[pivot]->flow;

	swap = arr[first];
	arr[first] = arr[pivot];
	arr[pivot] = swap;

	left = (first+1);

	while (left < right)
	{
		if (arr[left]->flow < pivotval)
		{
			swap = arr[left];
			arr[left] = arr[right];
			arr[right] = swap;
			-- right;
		}
		else
		{
			++ left;
		}
	}

	swap = arr[first];
	arr[first] = arr[left];
	arr[left] = swap;

	if (first < (left-1))
	{
		quickSort (arr, first, (left-1));
	}

	if ((left+1) < last)
	{
		quickSort (arr, (left+1), last);
	}
}

static void
sort (Node * current)
{
	if (current->numOutOfTree > 1)
	{
		quickSort (current->outOfTree, 0, (current->numOutOfTree-1));
	}
}

static void
minisort (Node *current)
{
	Arc *temp = current->outOfTree[current->nextArc];
	int i, size = current->numOutOfTree, tempflow = temp->flow;

	for(i=current->nextArc+1; ((i<size) && (tempflow < current->outOfTree[i]->flow)); ++i)
	{
		current->outOfTree[i-1] = current->outOfTree[i];
	}
	current->outOfTree[i-1] = temp;
}

static void
decompose (Node *excessNode, const int source, int *iteration)
{
	Node *current = excessNode;
	Arc *tempArc;
    //int bottleneck = excessNode->excess;
    float bottleneck = excessNode->excess;

	for ( ;(current - adjacencyList+1 != source) && (current->visited < (*iteration));
				current = &adjacencyList[tempArc->from])
	{
		current->visited = (*iteration);
		tempArc = current->outOfTree[current->nextArc];

		if (tempArc->flow < bottleneck)
		{
			bottleneck = tempArc->flow;
		}
	}

	if (current - adjacencyList+1 == source)
	{
		excessNode->excess -= bottleneck;
		current = excessNode;

		while (current - adjacencyList+1 != source)
		{
			tempArc = current->outOfTree[current->nextArc];
			tempArc->flow -= bottleneck;

			if (tempArc->flow)
			{
				minisort(current);
			}
			else
			{
				++ current->nextArc;
			}
			current = &adjacencyList[tempArc->from];
		}
		return;
	}

	++ (*iteration);

	bottleneck = current->outOfTree[current->nextArc]->flow;

	while (current->visited < (*iteration))
	{
		current->visited = (*iteration);
		tempArc = current->outOfTree[current->nextArc];

		if (tempArc->flow < bottleneck)
		{
			bottleneck = tempArc->flow;
		}
		current = &adjacencyList[tempArc->from];
	}

	++ (*iteration);

	while (current->visited < (*iteration))
	{
		current->visited = (*iteration);

		tempArc = current->outOfTree[current->nextArc];
		tempArc->flow -= bottleneck;

		if (tempArc->flow)
		{
			minisort(current);
			current = &adjacencyList[tempArc->from];
		}
		else
		{
			++ current->nextArc;
			current = &adjacencyList[tempArc->from];
		}
	}
}

static void
recoverFlow (void)
{
	int i, j, iteration = 1;
	Arc *tempArc;
	Node *tempNode;

	for (i=0; i<adjacencyList[sink].numOutOfTree; ++i)
	{
		tempArc = adjacencyList[sink].outOfTree[i];
		if (adjacencyList[tempArc->from].excess < 0)
		{
			tempArc->flow -= (float) (-1*adjacencyList[tempArc->from].excess);
			adjacencyList[tempArc->from].excess = 0;
		}
	}

	for (i=0; i<adjacencyList[source].numOutOfTree; ++i)
	{
		tempArc = adjacencyList[source].outOfTree[i];
		addOutOfTreeNode (&adjacencyList[tempArc->to], tempArc);
	}

	adjacencyList[source].excess = 0;
	adjacencyList[sink].excess = 0;

	for (i=0; i<numNodes; ++i)
	{
		tempNode = &adjacencyList[i];

		if ((i == (source)) || (i == (sink)))
		{
			continue;
		}

		if (tempNode->label >= numNodes)
		{
			tempNode->nextArc = 0;
			if ((tempNode->parent) && (tempNode->arcToParent->flow))
			{
				addOutOfTreeNode (&adjacencyList[tempNode->arcToParent->to], tempNode->arcToParent);
			}

			for (j=0; j<tempNode->numOutOfTree; ++j)
			{
				if (!tempNode->outOfTree[j]->flow)
				{
					-- tempNode->numOutOfTree;
					tempNode->outOfTree[j] = tempNode->outOfTree[tempNode->numOutOfTree];
					-- j;
				}
			}

			sort(tempNode);
		}
	}

	for (i=0; i<numNodes; ++i)
	{
		tempNode = &adjacencyList[i];
		while (tempNode->excess > 0)
		{
			++ iteration;
			decompose(tempNode, source, &iteration);
		}
	}
}


static void
outputBreakpoints (int ** breakpoints)
{
    int* breakpointsPointer;
    if ((breakpointsPointer = (int *)malloc((numNodes) * sizeof(int))) == NULL)
  	{
  		printf("Could not allocate memory.\n");
  		exit(0);
  	}
    for (int i=0; i<numNodes; ++i)
    {
        breakpointsPointer[i] = adjacencyList[i].breakpoint;
    }
    *breakpoints = breakpointsPointer;
}

void libfree(void* p)
{
  //printf("c Freeing the breakpoints\n");
	free(p);
}

static void
freeMemory (void)
{
	int i;

	for (i=0; i<numNodes; ++i)
	{
		freeRoot (&strongRoots[i]);
	}

	free (strongRoots);

	for (i=0; i<numNodes; ++i)
	{
		if (adjacencyList[i].outOfTree)
		{
			free (adjacencyList[i].outOfTree);
		}
	}

	free (adjacencyList);

	free (labelCount);

	/*
  for (int i=0; i<numArcs; ++i)
  {
    free (arcList[i].capacities);
  }

	*/

	free (arcList);
}

void
simparam_solve(int origNumNodes, int origNumArcs, int numLambda,
	float * eds, int sourceID, int sinkID, float * lambdas, int ** breakpoints)
{
	source_set_size = 1;
	highestStrongLabel=1;
	float theinitime = timer();
	//printf ("\nc Pseudoflow algorithm for parametric min cut (version 1.0)\n");
	readData(origNumNodes, origNumArcs, numLambda, eds, sourceID, sinkID, lambdas);
	//printf ("c PseudoFlow Finished preparing data in %.3lf seconds\n", (timer()-theinitime)); fflush (stdout);
  float thetime = timer();
	simpleInitialization ();
  //printf("c Finished simple Initialization in %.3lf seconds\n", (timer()-theinitime)); fflush (stdout);
  thetime = timer();
	pseudoflowPhase1 ();

  //printf("c Finished Phase 1 in %.3lf seconds\n", (timer()-theinitime)); fflush (stdout);
#ifdef RECOVER_FLOW
	recoverFlow();
	checkOptimality ();
#endif

	//printf ("c Number of nodes     : %d\n", numNodes);
	//printf ("c Number of arcs      : %d\n", numArcs);
#ifdef STATS
	printf ("c Number of arc scans : %lld\n", numArcScans);
	printf ("c Number of mergers   : %d\n", numMergers);
	printf ("c Number of pushes    : %lld\n", numPushes);
	printf ("c Number of relabels  : %d\n", numRelabels);
	printf ("c Number of gaps      : %d\n", numGaps);
#endif

		thetime = timer();
    outputBreakpoints (breakpoints);
    //printf("c PseudoFlow Finished create output file in %.3lf seconds\n", (timer()-thetime));
	//	printf("c PseudoFlow completed the algorithm in %.3lf seconds\n", (timer()-theinitime));

	freeMemory ();

	return;
}
