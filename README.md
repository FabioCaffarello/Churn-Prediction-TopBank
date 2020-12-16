# Churn-Prediction-TopBank
Predict whether a fictitious bank customer will be a churn or not


# Context


TopBank is a large banking services company operating mainly in European countries offering financial products, from bank accounts to investments, as well as some types of insurance and investment products. The company sells banking services to its customers through physical branches and an online portal, so its business model is of the service type.

The company's main product is a bank account, in which the client can deposit his salary, make withdrawals, deposits and transfer to other accounts. This bank account has no cost for the client and is valid for 12 months, i.e. the client needs to renew the contract of this account to continue using for the next 12 months.

According to the TopBank Analytics team, each client who holds this bank account returns a monetary value of 15% of their estimated salary, if this is less than the average, and 20% if this salary is higher than the average, during the current period of their account. This value is calculated annually. 

For example, if a client's monthly salary is EUR 1,000.00 and the average of all bank salaries is EUR 800. The company, therefore, invoices EUR 200 annually with this client. If this client has been in the bank for 10 years, the company has already invoiced EUR 2,000.00 with its transactions and use of the account. 

In recent months, the Analytics team realized that the rate of clients canceling their accounts and leaving the bank, reached unprecedented numbers in the company. Concerned with the increase of this rate, the team planned an action plan to decrease the customer evasion rate.

Concerned about the drop in this metric, TopBottom's Analytics team hired you as a Data Science consultant to create an action plan to reduce client evasion, i.e., prevent the client from cancelling their contract and not renewing it for another 12 months. This evasion, in business metrics, is known as Churn.

In general, Churn is a metric that indicates the number of customers who have cancelled their contract or stopped purchasing their product in a certain period of time. For example, customers who have cancelled their service contract or after its expiration have not renewed, are customers considered in churn.

Another example would be customers who have not made a purchase for more than 60 days. These clients can be considered churn clients until a purchase is made. The 60-day period is totally arbitrary and varies between companies.

# Data

## Data Collection

- The data was collected from a dataset provied by Merven Torkan in Kaggle.
- **Link:** <a href="https://www.kaggle.com/mervetorkan/churndataset">Churn Dataset</a>

## Data Information

- RowNumber:       Corresponds to the record (row) number and has no effect on the output.
- CustomerId:      Contains random values and has no effect on customer leaving the bank.
- Surname:         The surname of a customer has no impact on their decision to leave the bank.
- CreditScore:     Can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- Geography:       A customer’s location can affect their decision to leave the bank.
- Gender:          It’s interesting to explore whether gender plays a role in a customer leaving the bank.
- Age:             This is certainly relevant, since older customers are less likely to leave their bank than younger ones.
- Tenure:          Refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
- Balance:         Also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
- NumOfProducts:   Refers to the number of products that a customer has purchased through the bank.
- HasCrCard:       Denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
- IsActiveMember:  Active customers are less likely to leave the bank.
- EstimatedSalary: As with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- Exited:          Whether or not the customer left the bank. (0=No,1=Yes)