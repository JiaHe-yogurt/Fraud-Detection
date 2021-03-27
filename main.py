############# Question 1
## upload dataset
data = []
with open('your path/transactions.txt') as f:
    for line in f:
        data.append(json.loads(line))
## convert data into array for further analyze
df = pd.DataFrame(data)
## check data information
# replace empty string with np.nan
df.replace('', np.nan, inplace=True)
# find counts for all  fields and plot
Count=df.count()
Count.plot.bar()
sns.set(style='whitegrid')
plt.axhline(y=Count.max(),linewidth=1, color='r')
# find columns with missing value
df.columns[np.where(Count<Count.max())[0]]
# caculate the number of null record in fields with missing entry
df.shape[0]-df.count()[np.where(Count<Count.max())[0]]
# complete missing
NonNull=df.count()
df.columns[np.where(NonNull==0)[0]]
# drop fields with complete empty entry
todrop=df.iloc[:,np.where(NonNull==0)[0]]
df1=df.drop(columns=todrop)
# descriptive statistics for all fields
print(df1.describe(include='all'))
# convert jsontimestamp to datetime in python
## this may take some time
for i in range(0, len(df1)):
    a=dateutil.parser.parse(df1['transactionDateTime'][i])
    df1['transactionDateTime'][i]=a

## check ratio between Fraud and nonFraud
freq=df1['isFraud'].value_counts()
ratio = np.round(freq / len(df1.index), 3)
print(f'Ratio of fraud cases: {ratio[1]}\nRatio of non-fraud cases: {ratio[0]}')



########## Question 2
# plot a histogram of transactionAmount
plt.figure(figsize=(12,4), dpi=80)
sns.distplot(df1['transactionAmount'], bins=300, kde=False)
plt.ylabel('Count')
plt.title('Transaction Amounts')
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)
mpl.style.use('ggplot')
sns.set(style='whitegrid')
## plot boxplot
plt.figure(figsize=(12,4), dpi=80)
sns.boxplot(df1['transactionAmount'])
plt.title('Transaction Amounts')

## box-cox transform
df1['transactionAmount']= df1['transactionAmount'] + 1e-9 # Shift all amounts by 1e-9
df1['transactionAmount'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(df1['transactionAmount'], alpha=0.01)
plt.figure(figsize=(12,4), dpi=80)
sns.distplot(df1['transactionAmount'], kde=False)
plt.xlabel('Transformed Transaction Amounts')
plt.ylabel('Count')
plt.title('Transaction Amounts (Box-Cox Transformed)')

# bar plot of categorical var
chart=sns.catplot(x="merchantCategoryCode", kind="count", palette="ch:.25",data=df1)
chart.set_xticklabels(rotation=90)
chart.set_xlabels("X Label",fontsize=10)
plt.title('Bar Plot of merchantCategoryCode')

chart=sns.catplot(x="transactionType", kind="count", palette="ch:.25",data=df1)
plt.title('Bar Plot of transactionType')




####### Question 3
######## find reversal transaction
rev_tra=df1['transactionType'].value_counts()
## estimate the total dollar amount for reversal transaction
print('Total dollar amount for reversal transaction is:', np.sum(df1[df1['transactionType']=='REVERSAL']['transactionAmount']))

## find multi wipe
# mark the duplicate payments and mark per customer if the difference is less then 1 minutes
df1.sort_values(['customerId', 'transactionDateTime'], inplace=True)
m1 = df1.groupby('customerId', sort=False)['transactionAmount'].apply(lambda x: x.duplicated())
m2 = df1.groupby('customerId', sort=False)['transactionDateTime'].diff() <= pd.Timedelta(1, unit='minutes')
df1['Duplicated?'] = np.where(m1 & m2, 'Yes', 'No')
df1['Duplicated?'].value_counts()
print('Total dollar amount for multiswipe transaction is:', np.sum(df1[df1['Duplicated?']=='Yes']['transactionAmount']))

##### Question 4
# drop duplicated and reversal trans
df1.drop(df1[(df1['transactionType'] =='REVERSAL')| (df1['Duplicated?']=='Yes') ].index, inplace = True)
# drop record with missing input
data=df1.dropna()
N_idx, P_idx =data.index[data['isFraud'] == False].tolist(), data.index[data['isFraud'] == True].tolist()
Seed=1
N_e, P_e =data.loc[N_idx].sample(2000,random_state=Seed), data.loc[P_idx].sample(2000,random_state=Seed)
data=pd.concat([P_e,N_e])
data=data.reset_index(drop=True) #'
X = data[[ 'availableMoney', 'transactionAmount', 'acqCountry', 'merchantCategoryCode',
     'posEntryMode', 'transactionType', 'posConditionCode','currentBalance', 'cardPresent', 'expirationDateKeyInMatch']]
## change the dtype of following columns to categorical
cols=[  'acqCountry',  'posEntryMode', 'merchantCategoryCode',
      'transactionType','posConditionCode', 'cardPresent', 'expirationDateKeyInMatch']
X[cols]=X[cols].astype('category')
X=pd.get_dummies(X[cols])
y = data['isFraud']




# three fold cross validation
## set seed for reproducibility
seed = 2
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.
## NN
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
NN_recall, NN_accu, LR_recall, LR_accu, RF_recall,RF_accu =[],[],[],[],[],[]
for train, test in kfold.split(X, y):
  model = Sequential()
  model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  opt = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy'])
  model.fit(X.loc[train], encoded_y[train], epochs=150, batch_size=200, class_weight={0:1,1:1.5}, verbose=0)
  nn_pred= model.predict_classes(X.loc[test])
  print(confusion_matrix(y_true=encoded_y[test], y_pred=nn_pred))
  NN_recall.append(recall_score(y[test], nn_pred)), NN_accu.append(accuracy_score(y[test], nn_pred))

print('Neural Network Recall: ' + str(np.mean(NN_recall)),'Accuracy: ' +str(np.mean(NN_accu))) # results may vary during to randomness

## Logistic regression
for train, test in kfold.split(X, y):
  lr_model = LogisticRegression(random_state=0, solver='liblinear',class_weight={False: 1, True: 2}).fit(X.loc[train], y[train])
  lrpred = lr_model.predict(X.loc[test])
  print(confusion_matrix(y_true=y[test], y_pred=lrpred,labels=[False,True]))
  rc, accu =recall_score(y[test],lrpred),accuracy_score(y[test],lrpred)
  LR_recall.append(rc), LR_accu.append(accu)
print('Logistic Regression Recall: ' + str(np.mean(LR_recall)),'Accuracy: ' +str(np.mean(LR_accu)))  # results may vary during to randomness

##  Random forest
for train, test in kfold.split(X, y):
  rf_model = RandomForestClassifier(n_estimators=100, class_weight= {False: 1, True: 2},random_state=0).fit(X.loc[train], y[train])
  rfpred= rf_model.predict(X.loc[test])
  print(confusion_matrix(y_true=y[test], y_pred=rfpred,labels=[False,True]))
  rc, accu =recall_score(y[test],rfpred),accuracy_score(y[test],rfpred)
  RF_recall.append( rc),RF_accu.append(accu)
print('Random Forest Recall: ' + str(np.mean(RF_recall)),'Accuracy: ' +str(np.mean(RF_accu)))  # results may vary during to randomness
