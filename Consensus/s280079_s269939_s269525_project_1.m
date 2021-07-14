%% Mar 4, 2021 -- Consensus "Hello world" -- S. M. Fosson%%

clear all
close all
clc

%% grid points
p = 100;
Rm = zeros(10,10);
%number of agents/nodes
n = 25;
%measurements
m = 1;
lambda = 1e-4;
tau = 0.7;
%Set grid to 1 for a grid topology
%Set grid to 0 for a random topology
grid = 0;

expN = 20;
count = 50;
IST_count = zeros(count,2);
DIST_count = zeros(count,2);

for c=1:count

isConnected = 0;
while(isConnected == 0)

    
Q = zeros(n,n);
inD = zeros(n,1);

%topology generation
%random
    if(grid == 0) %random topology
        r = 4;
        x_0 = rand(n,1).*10;
        y_0 = rand(n,1).*10;
    else    %grid topology
        r = 2;
        x_0 = [1;3;5;7;9;1;3;5;7;9;1;3;5;7;9;1;3;5;7;9;1;3;5;7;9];
        y_0 = [1;1;1;1;1;3;3;3;3;3;5;5;5;5;5;7;7;7;7;7;9;9;9;9;9];
    end

% Qmatrix and inDegree initialization
for i=1:n
    for j =1:n
        dist = sqrt((x_0(i)-x_0(j))^2 + (y_0(i)-y_0(j))^2);
        if ( dist <= r )
            Q(i,j) = 1;
           inD(i)=inD(i)+1;
        end
    end
end
%Conncetivity check 
G = graph(Q);
isConnected = all(conncomp(G) == 1);
end
% Metropolis-Hastings weights for Q matrix
for i=1:n
    for j =1:n
       if(i ~= j)
           if( Q(i,j) == 1 ) 
           Q(i,j) = 1/(max(inD(i),inD(j))+1);
           end
       end
    end
    Q(i,i) = 2-sum(Q(i,:));
end
%%Get essential spectral radius/second largest eigenvalue
esr = sort(abs(eig(Q)),'descend');
esr = esr(2);
IST_count(c,2) = esr;
DIST_count(c,2) = esr;

%% Training

%Fingerprinting RSS using RSS function
%costx and costy are the center points of i^th cell
A = zeros(m*n,p);
for i=1:p
    costx = mod(i-1,10)+0.5;
    costy = floor((i-1)/10)+0.5;
    for j=1:n
        dist = sqrt((x_0(j)-costx)^2 + (y_0(j)-costy)^2);
        A(j,i) = RSS(dist);
    end
    
end

%% Feng Orthogonalization

A_orth = (orth(A'))';
A_pinv = pinv(A);
A_mod = A_orth*A_pinv*A;

%mutual_coherence(A)
%mutual_coherence(A_orth)
%mutual_coherence(A_mod)

%% Runtime
% trgt_x = rand*10;
% trgt_y = rand*10;
%Dummy target for testing purposes
successIST = 0;
failIST = 0;
successDIST = 0;
failDIST = 0;
failDistance = zeros(expN,5);
IST_stop = zeros(expN,1);
DIST_stop = zeros(expN,1);

%% Multiple Experiments
for exp=1:expN
    
trgt_x = rand*10;
trgt_y = rand*10;


y = measure(trgt_x,trgt_y,m,n,x_0,y_0);
y_sign = A_orth*A_pinv*y;

%% IST -DIST
% get starting x input for the values
[x_IST,IST_stop(exp)] = outerIST(A_orth,y_sign,lambda,tau,p);
[x_DIST,DIST_stop(exp)] = outerDIST(A_orth,y_sign,lambda,tau,Q,n,p,zeros(p,n),0);

%For IST take maximum of all predictions (1 prediction per agent)
[M,I1] = max(x_IST);
%For DIST take maximum over the mean of all predictions ( n prediction per
%cell
[M,I2] = max(mean(x_DIST,2));
%actual cell 
pos = floor(trgt_y)*10+floor(trgt_x)+1;

%success percentage calculation
if( I1 == pos)
    successIST = successIST+1;
else
    failIST = failIST+1;
    failDistance(exp,1) = pos;
    failDistance(exp,2) = I1;
    failDistance(exp,3) = norm([mod(I1-1,10)+0.5;floor((I1-1)/10)+0.5]-[trgt_x;trgt_y]);
end

if( I2 == pos)
    successDIST = successDIST+1;
else
    failDIST = failDIST+1;
    failDistance(exp,1) = pos;
    failDistance(exp,4) = I2;
    failDistance(exp,5) = norm([mod(I2-1,10)+0.5;floor((I2-1)/10)+0.5]-[trgt_x;trgt_y]);
end


end
successIST = successIST/(successIST+failIST)*100;
successDIST = successDIST/(successDIST+failDIST)*100;
IST_count(c,1) = mean(IST_stop);
DIST_count(c,1) = mean(DIST_stop);
end


%% O-DIST

Tmax = 10;
experiments_tot_dist = zeros(expN,1);
random_dir = [-1;0;1];
path=zeros(Tmax, expN);
true_path = zeros(Tmax, expN);

for exp = 1:expN

    curr_dist=zeros(Tmax,1);
    tot_dist=0;

    trgt_x = floor(rand*10) + 0.5;
    trgt_y = floor(rand*10) + 0.5;

    x_prec = zeros(p,n);
    for t=1:Tmax

        is_bound = 0;
        %%Target cell update with boundary checking
        while(is_bound == 0) 
            move_x = trgt_x + randsample(random_dir, 1);
            move_y = trgt_y + randsample(random_dir, 1);

            if(move_x < 0 || move_y < 0 || move_x > 10 || move_y > 10 || (move_x == trgt_x && move_y == trgt_y))
                continue
            else
                is_bound = 1;
            end
        end

        trgt_x = move_x;
        trgt_y = move_y;

        %measure target 
        y = measure(trgt_x,trgt_y,m,n,x_0,y_0);
        y_sign = A_orth*A_pinv*y;

        %DIST function with Tmax variable set != 0 for stopped version
        [x,temp]=outerDIST(A_orth,y_sign,lambda,tau,Q,n,p,x_prec,1000);

        %experiment measurement

        x_prec=x;
        [M,I] = max(mean(x_prec,2));
        path(t, exp) = I;
        true_path(t, exp)= floor(trgt_y)*10+floor(trgt_x)+1;
        est_x = mod(I-1,10)+0.5;
        est_y = floor((I-1)/10)+0.5;
        curr_dist(t)= norm([est_x;est_y]-[trgt_x;trgt_y]);
        tot_dist=tot_dist+curr_dist(t);

    end

    % saving the exp-th total distance error
    experiments_tot_dist(exp) = tot_dist;

end
%% Sensor display

colors=rand(n,3);
figure(2);
for i = 1:n
        plot(x_0(i),y_0(i),'.r','MarkerSize', 20,'Color',[colors(i,:)])
        axis([0 10 0 10])
        hold on
end
axis([0 10 0 10])
hold off


%% Rss Model


function r = RSS(d)
    P_t = 25;
    sigma = 0.5;
    mu = 0;
    noise = sigma*randn+mu;
    if( d <= 8 ) 
        r = P_t - 40.2 - 20*log10(d) + noise;
    else 
        r = P_t - 58.5 - 33*log10(d) + noise;
    end
end

%% Outer IST
%Wrapper function for IST step
function [x,t] = outerIST(A,y,lambda,tau,p)
x_prec = zeros(p,1);

    for t =1:1000000
        x = IST(A,y,lambda,tau,x_prec);
        %pruning for loop when converging to lower values
        if (norm(x-x_prec) < 1e-3)
            x_prec = x;
            break;
        end
        x_prec = x;
    end
end

function x = IST(A,y,lambda,tau,x_prec)
        x = x_prec + (tau.*A.')*(y-A*x_prec);
        for i=1:length(x_prec)
            x_c = ST(lambda,x(i));
            x(i) = x_c;
        end   
end

%% Outer DIST
%Wrapper function for the DIST and O-DIST algorithms
%here Tmax is used both in DIST and ODIST to add the stopped version in the
%case of the ODIST
function [x,t] = outerDIST(A,y,lambda,tau,Q,n,p,x_prec,Tmax)
isOnline = (Tmax > 0 );
for t =1:1000000
        %run DIST at each time step for every sensor
        x = DIST(A,y,lambda,tau,n,p,Q,x_prec);
        
        %pruning for loop 
        if (norm(x-x_prec) < 1e-5 || (isOnline && t > Tmax) )
            x_prec = x;
            break;
        end
        x_prec = x;
end

end

%%DIST
%inner DIST algorithm
function x = DIST(A,y,lambda,tau,n,p,Q, x_prec)
x=x_prec;
for i=1:n
    xi_new = (Q(i,:)*x')' +(tau.*A(i,:).')*(y(i)-A(i,:)*x(:,i));
                %Qij*xj     %tau*Ai         %yi   Ai*xi
    for j=1:p
      xi_new(j) = ST(lambda,xi_new(j));
    end 
    x(:,i) = xi_new;
end
end



%% Soft Thresholding function 
function a = ST(lambda,x)
    if( abs(x) <= lambda ) 
        a = 0;
    else
        a = x-sign(x)*lambda;
    end
end

%% Measurement for fingerprinting and tracking
function y=measure(trgt_x,trgt_y,m,n,x_0,y_0)
y = zeros(n,m);

    %Measurements using same function as training
    for j=1:m
        for i=1:n
            dist = sqrt((x_0(i)-trgt_x)^2 + (y_0(i)-trgt_y)^2);
            y(i,j) = RSS(dist);
        end
    end
end