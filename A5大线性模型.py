class Node(object):
    def __init__(self,val,next_=None):
        self.val=val
        self.next=next_
class LinkList(object):
    def __init__(self):
        self.head=Node(None) 
    def init_list(self,l):
        
        p=self.head 
        for i in l:
            p.next=Node(i)
            p=p.next
    def show(self):
        p=self.head.next
        while p:
            print(p.val,end='')
            p=p.next
        print()
    def append(self,item):
        p=self.head
        while p.next is not None:
            p=p.next
        p.next=Node(item)
    def get_length(self):
        n=0
        p=self.head
        while p.next is not None:
            n+=1
            p=p.next
        return n 
    def is_empty(self):
        if self.get_length()==0:
            return True
        else:
            return False
    def clear(self):
        self.head.next=None
    def get_item(self,index):
        p=self.head.next
        i=0
        while i <index and p is not None:
            p=p.next
            i+=1
        if p is None:
            raise IndexError('linklist index out of range')
        else:
            return p.val
    def insert(self,index,item):
        if index<0 or index>self.get_length():
            raise IndexError('linklist index out of range')
        
        p=self.head
        i=0
        while i<index:
            p=p.next
            i+=1
        node=Node(item)
        node.next=p.next
        p.next=node
    def delete(self,item):
        p=self.head
        while p.next:
            if p.next.val==item:
                p.next=p.next.next
                break
            p=p.next
        else:
            raise ValueError('x not in list')
            
class StackError(Exception):
    pass
class SStack(object):
    def __init__(self):
        #约定列表的最后一个元素为栈顶
        self._elems=[]#_受保护的，类或者子类中调用(认为)element
    def top(self):
        if not self._elems:
            raise StackError("stack is empty")
        return self._elems[-1]
    def is_empty(self):
        return self._elems==[]
    def push(self,elem):
        self._elems.append(elem)
    def pop(self):
        if not self._elems:
            raise StackError("stack is empty")
        return self._elems.pop()
        
class StackError(Exception):
    pass
class Node:
    def __init__(self,val,next_=None):
        self.val=val
        self.next=next_
class LStack:
    def __init__(self):
        self._top=None
    def is_empty(self):
        return self._top is None
    def push(self,elem):
        #node=Node(elem)
        #node.next=self._top
        #self._top=node
        self._top=Node(elem,self._top)
    def pop(self):
        if self._top is None:
            raise StackError("stack is empty")
        p=self._top
        self._top=p.next
        return p.val
    def top(self):
        if self._top is None:
            raise StackError("stack is empty")
        return self._top.val
    
class QueueError(Exception):
    pass
# 完成队列操作
class SQueue:
    def __init__(self):
        self._elems = []  # 使用列表存储队列数据

    def is_empty(self):
        return self._elems == []

    # 　入队
    def enqueue(self, elem):
        self._elems.append(elem)

    # 出队
    def dequeue(self):
        if not self._elems:
            raise QueueError("Queue is empty")
        return self._elems.pop(0)

# 　队列异常
class QueueError(Exception):
    pass
# 队列结点类
class Node(object):
    def __init__(self, val, next=None):
        self.val = val  # 有用数据
        self.next = next

#　链式队列方法实现
class LQueue:
    def __init__(self):
        self.front = self.rear = Node(None)

    def is_empty(self):
        return self.front == self.rear

    #　入队
    def enqueue(self,elem):
        self.rear.next = Node(elem)
        self.rear = self.rear.next

    #　出队
    def dequeue(self):
        if self.front == self.rear:
            raise QueueError("Queue is empty")
        self.front = self.front.next
        return self.front.val

    #　清空队列
    def clear(self):
        self.front = self.rear