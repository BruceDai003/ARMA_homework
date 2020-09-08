%% Problem 2
[price, txt] = xlsread('GSPC.csv');
% price = csvread('sp500.csv', 1, 1);
dates = datetime(txt(2:end, 1), 'InputFormat', 'yyyy/MM/dd');
adj_close = price(:, 5);
open_price = price(:, 1);
ret = (log(adj_close) - log(open_price)) * 100;

% 1) Estimate a pool of ARMA(p, q) models
tp = 6;
tq = 6;

% Save the results in to a 6x6 matrix( row for p and column for q)
MSE     = zeros(tp,tq); % in-sample MSE
ssq     = zeros(tp,tq); % s^2
loglik  = zeros(tp,tq); % log-likelihood 
resids  = cell(tp,tq);  % cell data type
k       = zeros(tp,tq); % number of parameters


T = numel(ret);       % sample size
for p = 0:tp-1
  for q = 0:tq-1
    mdl = arima(p,0,q);
    [estmdl, ~, ~] = estimate(mdl, ret, 'display', 'off');
    [resid, ~, loglik(p+1,q+1)] = infer(estmdl, ret);
    
    % number of parameters in the model
    k(p+1,q+1) = p + q + 1;
    % MSE and s^2
    MSE(p+1,q+1) = mean(resid.^2);                     
    ssq(p+1,q+1) = T * MSE(p+1,q+1) ./ (T-k(p+1,q+1)); 
    % save residuals for each each in the cell entry
    resids{p+1,q+1} = resid;
  end 
end

% Compute AIC, HQIC, BIC
aic2 = exp(2*k/T) .* MSE;
hqic = (log(T)).^(2*k/T).* MSE; 
bic2 = (sqrt(T)).^(2*k/T).* MSE; 

[r, c] = find(aic2 == min(min(aic2)));
best_p = r-1;
best_q = c-1;
fprintf('Best model - Textbook AIC: ARMA(%d, %d)\n', best_p, best_q);

[r, c] = find(bic2 == min(min(bic2)));
best_p = r-1;
best_q = c-1;
fprintf('Best model - Textbook BIC: ARMA(%d, %d)\n', best_p, best_q);

[r, c] = find(hqic == min(min(hqic)));
best_p = r-1;
best_q = c-1;
fprintf('Best model - HQIC: ARMA(%d, %d)\n', best_p, best_q);

% ii) Obtain residuals selected by BIC, estimate GARCH(1, 1)
mdl = arima(0, 0, 0);
estmdl = estimate(mdl, ret);
resid = infer(estmdl, ret);

% GARCH(1, 1)
mdlgarch = garch('GARCHLags', 1, 'ARCHLags', 1);
estgarchmdl = estimate(mdlgarch, resid);

% iii) Plot the estimated conditional volatility
% GARCH(1,1) parameter estimates
omega = estgarchmdl.Constant
alpha = estgarchmdl.ARCH{1}
beta  = estgarchmdl.GARCH{1}

sig2bar = omega/(1-alpha-beta);
sigsq = infer(estgarchmdl, resid, 'V0', sig2bar, 'E0', sqrt(sig2bar));

% Plot the estimated conditional volatility from GARCH(1,1) model
% conditional volatility 
vol = sqrt(sigsq);
vol = sqrt(252) * vol; % Annualized volatility
figure
box on
plot(vol)
grid on
ylabel('(%)')
title('GARCH conditional volatility of S&P500 continuously compounded returns')

%% Problem 3
premium_data = xlsread('term_premium.xlsx');
time_data = premium_data(:, 1);
premium = premium_data(:, 2);
% i) time-series plot
figure;
box on
plot(premium);
xlabel('year');
ylabel('term premium');
title('10-year zero coupon bond term preimum');


% ii) Estimate an ARMA(1, 1) model
sample = premium(1:364);
mdl = arima(1, 0, 1);
estmdl = estimate(mdl, sample);

% iii) One-step-ahead forecast
phi0 = estmdl.Constant;
phi1 = estmdl.AR{1};
theta = estmdl.MA{1};
resid = infer(estmdl, premium);
Y_t = phi0 + phi1 * sample(end) + theta * resid(364);

% iv) Compute Y from t = 0, 1, ..., T-1
resid0 = 0;
Y0 = phi0 / ( 1 - phi1);
Y_est = zeros(numel(premium), 1);
for i = 1:numel(premium)
    if i == 1
        Y_est(i) = phi0 + phi1 * Y0;
    else
        Y_est(i) = phi0 + phi1 * premium(i-1) + theta * resid(i-1);
    end

end

% Plot together
figure;
plot(premium);
hold on
plot( Y_est);
grid on;
legend('Real', 'Forecast');
xlabel('year');
ylabel('term premium');
title('10-year zero coupon bond term preimum');

% v) Plot forecast errors
forecast_errors = premium - Y_est;
forecast_errors = forecast_errors(365:end);
Y_est = Y_est(365:end);
figure;
plot(forecast_errors);
grid on;
xlabel('year');
ylabel('forecast error');
title('Forecast Error');

% vi) 
figure;
autocorr(forecast_errors);

[h, pvalue, stat, cv] = lbqtest(forecast_errors, 'lags', [5,10,20]);


% vii) MZ regress
T = numel(forecast_errors);
p = 2;
y = forecast_errors;
cons = ones(T, 1);
X = [cons, Y_est];
betahat = X\y;

% Extract individual parameter estimate
beta0hat = betahat(1);
beta1hat = betahat(2);
yhat = X * betahat;
ehat = y - yhat;
yd = y - mean(y);
R2 = 1 - (ehat' * ehat)/(yd' * yd);
R2adj = 1 - (T-1)/(T-p) * (1-R2);
sigsqhat = ehat' * ehat / T;   % estimate of the error variance 
Vhat     = sigsqhat * (X' * X /T)^(-1);

% Standard errors
var_beta0hat = Vhat(1,1)/T;
var_beta1hat = Vhat(2,2)/T;
se_beta0hat = sqrt(var_beta0hat);
se_beta1hat = sqrt(var_beta1hat);

% Joint test of two parameters
b = [0; 0];
chi2stat = T * (betahat - b)' * Vhat^(-1) * (betahat - b)
% obtain the two-sided 95% critical value for the chi2-test
alpha = 0.05; % significance level
q         = 1 - alpha;    % note that chi squared test is always one-sided 
cv_chi2_p = chi2inv(q, p) % q-quantile of  chi-squared distribution with degree of freedom p
rej = chi2stat > cv_chi2_p
pvalue = 1-chi2cdf(chi2stat, p)
% rej = 1 means reject the null hypothesis.

% viii) Random walk
% By taking the expectation, clealy the best one-step ahead forecast is for Yt+1|t is Yt.

% ix) Repeat steps for random walk
Y_est2 = premium(364:end-1);
forecast_errors2 = premium(365:end) - Y_est2;

% Plot forecast errors
figure;
plot(forecast_errors2);
grid on;
xlabel('year');
ylabel('forecast error');
title('Forecast Error');

% ACF 
figure;
autocorr(forecast_errors2);

[h, pvalue, stat, cv] = lbqtest(forecast_errors2, 'lags', [5,10,20]);

% MZ regress
T = numel(forecast_errors2);
p = 2;
y = forecast_errors2;
cons = ones(T, 1);
X = [cons, Y_est2];
betahat = X\y;

% Extract individual parameter estimate
beta0hat = betahat(1);
beta1hat = betahat(2);
yhat = X * betahat;
ehat = y - yhat;
yd = y - mean(y);
R2 = 1 - (ehat' * ehat)/(yd' * yd);
R2adj = 1 - (T-1)/(T-p) * (1-R2);
sigsqhat = ehat' * ehat / T;   % estimate of the error variance 
Vhat     = sigsqhat * (X' * X /T)^(-1);

% Standard errors
var_beta0hat = Vhat(1,1)/T;
var_beta1hat = Vhat(2,2)/T;
se_beta0hat = sqrt(var_beta0hat);
se_beta1hat = sqrt(var_beta1hat);

% Joint test of two parameters
b = [0; 0];
chi2stat = T * (betahat - b)' * Vhat^(-1) * (betahat - b)
% obtain the two-sided 95% critical value for the chi2-test
alpha = 0.05; % significance level
q         = 1 - alpha;    % note that chi squared test is always one-sided 
cv_chi2_p = chi2inv(q, p) % q-quantile of  chi-squared distribution with degree of freedom p
rej = chi2stat > cv_chi2_p
pvalue = 1-chi2cdf(chi2stat, p)
% rej = 1 means reject the null hypothesis.

% x) Compare ARMA(1, 1) and Random Walk by DM test
dt = forecast_errors.^2 - forecast_errors2.^2;

dt_mean = mean(dt);
% Use white noise errors
dt_std = std(dt);
DM_wn = dt_mean / dt_std;
alpha = 0.05; % significance level
mu = 0; sig = 1; q = 1-alpha/2;
cv = norminv(q, mu, sig); %  q-quantile of normal distribution N(mu, sig^2)
rej = abs(DM_wn) > cv
% rej = 0, can't reject the null hypothesis that the errors are equal

% Newy-West errors
L = 5;
y = dt(L+1:end);
X = [];
for i = 1:L
    X = [X, dt(L+1-i:end-i)];
end
mdl = fitlm(X, y);
% -- White standard error --
WCov = hac(mdl, 'type', 'HC', 'weights', 'HC0', 'display', 'off');
% -- Newey-West standard error -- 
maxlag = floor(4*(T/100)^(2/9));
NWCov = hac(mdl,'bandwidth',maxlag+1,'display','off');

r = mdl.Coefficients.Estimate(2:end);
IL = eye(L); % IL is an L by L identity matrix 
R = [zeros(L, 1), IL]; % add one zero column vector to the left of IL
display('White Noise:');
[h,pValue, stat, cv] = waldtest(r,R,WCov)
% Since h = 1,reject the hypothesis that the forecast errors are equal
% And since dt_mean >0, we know that the Random Walk erros are smaller,
% So random walk model is better here.
display('Newy-West errors:');
[h,pValue, stat, cv] = waldtest(r,R,NWCov)
% Since h = 1,reject the hypothesis that the forecast errors are equal
% And since dt_mean >0, we know that the Random Walk erros are smaller,
% So random walk model is better here.


%% Problem 6
% Historical Simulation
alp = 0.05;
m = 252;
sp = m + 1;
ep = length(ret);

VaR_HS = zeros(length(sp:ep), 1);
ES_HS = zeros(length(sp:ep), 1);
iter = 0;
for i = sp : ep
  iter = iter+1;
  hisdata = ret(i-m: i-1); 
  VaR_HS(iter) = VaR(hisdata, alp);   
end 

% Plot the VaR Forecasts by HS from 2011 to 2015
figure('visible', 'on')
box on
grid on
hold on
plot(VaR_HS, 'linewidth', 1.2)
hold off
ylabel('Percentage')
title('Historical estimation of 5%-VaR for daily S&P500 returns')

% Parametric GARCH Model
mean_ret = mean(ret);
resid = ret - mean_ret;
resid = resid(1:m); % In-sample residuals

vmdl = garch(1, 1);
estvmdl = estimate(vmdl, resid);
omega = estvmdl.Constant;
alpha = estvmdl.ARCH{1};
beta  = estvmdl.GARCH{1};

% Conditional variance 2010(in-sample)
sig2_insample = infer(estvmdl, resid);

% Complete conditional variance 2011-2015 (out-of-sample)
sig2_GARCH = zeros(numel(ret),1);
sig2_GARCH(1:numel(sig2_insample)) = sig2_insample;

for i = numel(sig2_insample)+1:numel(ret)
  sig2_GARCH(i) = omega + alpha * ret(i-1)^2 + beta * sig2_GARCH(i-1);
end 

% VaR/ES of the returns r(t) is a linear transformation of VaR/ES of a
% standard normal random variable using the location (conditional mean)
% estimate and scale (conditional volatility) estimate
VaR_stdnorm = -norminv(alp);
VaR_GARCH = -mean_ret + sqrt(sig2_GARCH) * VaR_stdnorm;

% Plot the VaR by GARCH
figure('visible', 'on')
box on
grid on
hold on
plot(VaR_GARCH(sp:ep), 'linewidth', 1.2)
hold off
ylabel('Percentage')
title('GARCH estimation of 5%-VaR for daily S&P500 returns')