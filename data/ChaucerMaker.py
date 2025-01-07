# get the Chaucer file

#turning each line into a list, learned from this website:
# https://stackoverflow.com/questions/44817068/how-do-i-read-a-file-and-convert-each-line-into-strings-and-see-if-any-of-the-st

# pulls up the Chaucer file
chaucer = open("Chaucer", "r")

# splits up each line into a new element in a list
data = chaucer.readlines()
data = [d.strip() for d in data]

# creating the ME and PDE lists
# I just need every line printed with out any preceding spaces, numbers, brackets, etc
me = []
pde = []

# eliminates the number and many spaces starting each ME sentence
def meCleaner(line):
    count = 0
    for i in line:
        count += 1
        # counts each index but only replaces the original line with the indices that start with a letter or parenthesis
        if i.isalpha() or i == '(':
            line = line[count - 1:]
            return line

# some of the lines have notes for additional translations in parenthesis, I think its best to get rid of these
# parenthesis and their contents because it makes the translation less literal. However, I still need to keep
# parentheses that Chaucer wrote
def txtMaker(me, pde):
    for i in range(len(me)):
        reject = True
        for j in me[i]:
            # if the ME has a parenthesis, don't delete it from the PDE
            if j == '(':
                reject = False
            # if Chaucer didn't write the parenthesis, get rid of it
            if reject:
                for l in pde[i]:
                    if l == '(':
                        # in order to delete list contents, I converted the sentence into a list, deleted the
                        # elements of the parentheses, then turned it back into a string.
                        eList = [l for l in pde[i]]
                        start = eList.index('(')
                        end = eList.index(')')
                        del eList[start:end + 2] # plus two to get rid of the extra spaces
                        pde[i] = ''.join(eList)
    return me, pde

# splits the dataset into ME and PDE
for i in range(len(data)):
    for j in data[i]:
        # all ME starts with a line number and its translation is always the next line
        if j.isnumeric():
            me.append(meCleaner(data[i]))
            pde.append(data[i + 1])
            break

# print(data)
# rewriting the cleaned datasets
me, pde = txtMaker(me, pde)

# print(len(me))


# splitting the dataset
# train should be 70%, and test and validation should each be 15%
# 15% of is 129, and 70% is 602

def split(dataset):
    train, validation, test = [], [], []
    for i in range(len(dataset)):
        # taking the first 70% for training
        if i < 602:
            train.append(dataset[i])
        # taking the next 15% for validation
        elif i > 602 and i <= 731:
            validation.append(dataset[i])
        # taking the remaining 15% for testing
        else:
            test.append(dataset[i])
    return train, validation, test

# inputting ME and PDE to split them
meTrain, meValidation, meTest = split(me)
pdeTrain, pdeValidation, pdeTest = split(pde)

# prints the datasets how we want them to put them into our files
def printer(dataset):
    for i in dataset:
        print(i)

printer(meTest)
print()
printer(pdeTest)
# print(len(meTrain))
# print(len(meValidation))
# print(len(meTest))
