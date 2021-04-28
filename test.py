from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []
class Domain:
    def __init__(self,_name,_label,_length,_numbers,_entropy):
        self.name = _name
        self.label = _label
        self.length=_length
        self.numbers=_numbers
        self.entropy=_entropy

    def returnData(self):
        return [self.length, self.numbers, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def entropy(s):
    _, counts = np.unique(list(s), return_counts=True)
    total = sum(counts)
    percent = list(map(lambda x: x / total, counts))
    return sum(-n * np.log(n) for n in percent)

def number(s):
    return sum(c.isdigit() for c in s)
 
def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",") 
            name = tokens[0]
            label = tokens[1]
            length= len(name)
            entropy1=entropy(name)
            numbers= number(name) 
            domainlist.append(Domain(name,label,length,numbers,entropy1))

def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix,labelList)
    file1=open("test.txt")
    file2=open("result.txt","w")
    for line in file1:
        line = line.strip()
        if line.startswith("#") or line =="":
            continue
        tokens_test = line.split(",")
        name_test = tokens_test[0]
        length_test = len(name_test)
        numbers_test= number(name_test)
        entropy_test = entropy(name_test)
        if clf.predict([[length_test,numbers_test,entropy_test]])==0:
            label_test1 = "notdga"
            content=line+","+label_test1
            file2.write(content)
            file2.write("\n")
        if clf.predict([[length_test,numbers_test,entropy_test]])==1:
            label_test2 = "dga"
            content=line+","+label_test2
            file2.write(content)
            file2.write("\n")
    file1.close
    file2.close

if __name__ == '__main__':
    main()
