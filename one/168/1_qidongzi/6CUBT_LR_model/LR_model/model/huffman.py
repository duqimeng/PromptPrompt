#/usr/bin/python
#-*-coding:utf-8-*-
label_count={"label1":300,"label2":200,"label3":100}

class Node(object):
	def __init__(self,name=None,value=None):
		self._name=name
		self._value=value
		self._left=None
		self._right=None

a=[Node(key,value) for key,value in label_count.items()]
while len(a)!=1:
	a.sort(key=lambda node:node._value,reverse=True)
	c=Node(value=(a[-1]._value+a[-2]._value))
	c._left=a.pop(-1)
	c._right=a.pop(-1)
	a.append(c)
root=a[0]
b=range(len(label_count))

for no in a:
	












class HuffmanTree(object):
	"""docstring for HuffmanTree"""
	def __init__(self, char_weights):
		self.a=[Node(part[0],part[1]) for part in char_weights]
		while len(self.a)!=1:
			self.a.sort(key=lambda node:node._value,reverse=True)
			c=Node(value=(self.a[-1]._value+self.a[-2]._value))
			c._left=self.a.pop(-1)
			c._right=self.a.pop(-1)
			self.a.append(c)
		self.root=self.a[0]
		#print len(self.a)
		self.b=range(len(char_weights))


	def get_code1(self):
		ret = {}
		for node in self.a:
			if isinstance(node Node):
				print "is"



	def pre(self,tree,length):
		
		node=tree
		
		if (not node):
			return
		elif node._name:
			print node._name +"_code:",
			#ret[node._name]=[]
			temp=[]
			for i in range(length):
				temp.append(self.b[i])
				print self.b[i],
				#ret[node._name].append(self.b[i])
			print "\n"
			return node._name,temp
		self.b[length]=0
		print "_____________"
		self.pre(node._left,length+1)
		self.b[length]=1
		self.pre(node._right,length+1)
		return ret
	def get_code(self):
		ret={}

		#return self.pre(self.root,0)

if __name__ == '__main__':
	char_weight=[("a",5),("b",4),("c",10),("d",8),("f",15),("g",2)]
	tree = HuffmanTree(char_weight)
	tree.get_code1()
	'''
	result=tree.pre(tree.root,0)
	print result
	for name,code in result.items():
		print name,code
	'''
	