import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.svm import SVR
import math
from scipy.optimize import minimize

def minimize_distance(var, args):
  dis=0
  l = var[10]
  runs_remaining = args[0]
  overs_remaining = args[1]
  wickets_in_hand = args[2]
  for i in range(len(wickets_in_hand)):
    if(runs_remaining[i]>0):                                                                                # there is an outlier containg -1 for runs_remaining 
      if(wickets_in_hand[i] >0 and wickets_in_hand[i] < 11):
        pred = var[wickets_in_hand[i] -1] * (1.0 - math.exp(-1*l*overs_remaining[i]/var[wickets_in_hand[i] -1]))
        dis = dis + math.pow(pred - runs_remaining[i], 2)
#    if(runs_remaining[i]<0):
#      print(runs_remaining[i])
  for i in range(len(cooked_data_points)):                                                                  # adding loss of cooked points
    pred = var[9] * (1.0 - math.exp(-1*l*50/var[9]))
    dis = dis + math.pow(pred - cooked_data_points[i], 2)
  return dis

df1 = pd.read_csv("./../data/cricket.csv")                                                                  # reading the data
total_data= df1[['Innings','Over','Total.Runs','Wickets.in.Hand','Innings.Total.Runs','Total.Overs']]       # taking out imp columns
data_for_first_innings=total_data[(total_data['Innings']==1)]                                               # rows of our intrest
number_of_overs_completed=data_for_first_innings['Over'].values                                             # extracting values of each columns
overs_bowled_in_the_innings=data_for_first_innings['Total.Overs'].values
number_of_runs_scored_till_that_over=data_for_first_innings['Total.Runs'].values
wickets_in_hand=data_for_first_innings['Wickets.in.Hand'].values
number_of_runs_scored_in_innings=data_for_first_innings['Innings.Total.Runs'].values

data= df1[['Match','Innings','Total.Runs','Innings.Total.Runs']] 
reqd_row=data[(data['Innings']==1)]                                                                         # rows of our intrest to cook up extra points
match_number=reqd_row['Match'].values

cooked_data_points=[]    
seq=0
for i in range(len(number_of_runs_scored_in_innings)):
  if  (match_number[i] != seq):
    cooked_data_points.append(number_of_runs_scored_in_innings[i])
    seq=match_number[i]
#print(len(cooked_data_points))

var = [13.0, 25.0, 50.0, 80.0, 100.0, 140.0, 170.0, 210.0, 240.0, 280.0, 11]
number_of_runs_remaining=number_of_runs_scored_in_innings-number_of_runs_scored_till_that_over
number_of_overs_remaining=overs_bowled_in_the_innings - number_of_overs_completed
out = minimize(minimize_distance, var, args=[number_of_runs_remaining,number_of_overs_remaining, wickets_in_hand],method='L-BFGS-B')
loss=out['fun']
var_opt=out['x']

print("loss :- " + str(loss))
print("optimized variables = " + str(var_opt))

Z_N = var_opt[9] * ( 1- np.exp ( (- var_opt[10]*50) / var_opt[9]))
x_axis=[]
for i in range(51):
  x_axis.append(i)
for i in range(10):
  y_axis=[]
  for j in range(51):
    Z_n = var_opt[i] * ( 1- np.exp ( (- var_opt[10]*j) / var_opt[i]))
    y_axis.append(100*Z_n / Z_N)
  plt.plot(x_axis,y_axis,label='Z'+str(i+1))
  plt.legend()
  
plt.xlabel('Overs Remaining')
plt.ylabel('Resources Remaining')
plt.show()
#plt.savefig('/content/gdrive/My Drive/Colab Notebooks/data/abc.png')

# ******************************************************* redundent code *********************************************************************
#list1=[]
#list2=[]
#list3=[]
#list4=[]
#list5=[]
#list6=[]
#list7=[]
#list8=[]
#list9=[]
#list10=[]

#for index in range(len(number_of_runs_scored_till_that_over)):
#  if wickets_in_hand[index] == 1 :
#    list1.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#  
# elif wickets_in_hand[index] == 2 :
#    list2.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#  elif wickets_in_hand[index] == 3 :
#    list3.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
 # elif wickets_in_hand[index] == 4 :
#    list4.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
#  elif wickets_in_hand[index] == 5 :
 #   list5.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
#  elif wickets_in_hand[index] == 6 :
 #   list6.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
 # elif wickets_in_hand[index] == 7 :
  #  list7.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
 # elif wickets_in_hand[index] == 8 :
  #  list8.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
 # elif wickets_in_hand[index] == 9 :
  #  list9.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])
#
 # elif wickets_in_hand[index] == 10 :
  #  list10.append([overs_bowled_in_the_innings[index] - number_of_overs_completed[index],number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]])

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
#total_runs_scored_wicket_in_hand_matrix=np.zeros((50, 11), dtype=int)
#count_to_obtain_average_matrix=np.zeros((50, 11), dtype=int)
#useful_average_matrix=np.zeros((50, 11), dtype=int)

#total_runs_scored_wicket_in_hand_matrix=np.zeros((50, 11), dtype=int)
#count_to_obtain_average_matrix=np.zeros((50, 11), dtype=int)
#useful_average_matrix=np.zeros((50, 11), dtype=int)

#for index in range(len(number_of_runs_scored_till_that_over)): 
#  temp=number_of_runs_scored_in_innings[index]-number_of_runs_scored_till_that_over[index]
#  number_of_runs_remaining_temp = temp
#  temp1=overs_bowled_in_the_innings[index] - number_of_overs_completed[index]
#  number_of_overs_remaining_temp = temp1
#  wickets_in_hand_temp = wickets_in_hand[index]
#  j=wickets_in_hand_temp - 1
#  i=number_of_overs_remaining_temp - 1
#  total_runs_scored_wicket_in_hand_matrix[i][j] =  total_runs_scored_wicket_in_hand_matrix[i][j] + number_of_runs_remaining_temp
#  count_to_obtain_average_matrix[i][j] = count_to_obtain_average_matrix[i][j] + 1


#for index in range(len(number_of_runs_scored_till_that_over)):  
#  total_runs_scored_wicket_in_hand_matrix[49][9] = total_runs_scored_wicket_in_hand_matrix[49][9]  + number_of_runs_scored_in_innings[index]
#  count_to_obtain_average_matrix[49][9] = count_to_obtain_average_matrix[49][9]  + 1
  
#for i in range(50):
#  for j in range(11):
#    if count_to_obtain_average_matrix[i][j] !=0 :
#      useful_average_matrix[i][j]=total_runs_scored_wicket_in_hand_matrix[i][j] / count_to_obtain_average_matrix[i][j]

#for index in range(len(number_of_runs_scored_till_that_over)):

#for i in range (len(innings_values)):
#      if outs[i]==0:
#            if over_values[i]==39:
#                print("yy")
#            score.append(runs_values[i])
#            index.append(i)
#            overs.append(over_values[i])
#            innings_runs.append(total_innings_runs[i])
#print(score[:5])
#print(index[:5])
#print(overs[:5])
#print(innings_runs[:5])
#a=[]
#b=[]
#for i in range (len(overs)):
#    a.append(50-overs[i])
#    b.append(innings_runs[i]-score[i])
#svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#svr_lin = SVR(kernel='linear', C=100, gamma='auto')
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#               coef0=1)
#x=[]
#for i in a[:1000]:
#    x.append([i])
#svr_poly.fit(x,b)
#q=[]
#z=[]
#for i in range(1,51):
#    q.append(i)
#    z.append(svr_poly.predict([[i]]))
#plt.plot(q, svr_rbf.predict(q))