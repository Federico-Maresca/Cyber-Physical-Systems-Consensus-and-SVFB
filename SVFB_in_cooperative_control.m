close all;
clear all;

%% MAGLEV agent model
A = [0 1; 880.87 0];
B = [0; -9.9453];
C = [708.27 0];
D = 0;
%Initial conditions
random = 1;
if (random == 1)
x0 = [((rand(1,6)*4)-1)/C(1,1);rand(1,6)/707.27]; %agent
else
x0 = [0,0];    
end

x1 = [0;1/707.27]; %leader

%Poles for steady-state reference
pconst = [0 -1];
psine = [-j +j];
pramp = [0 0];
p = psine;
%%Parameter GRID-SEARCH
%use cellcomb as combination of values
% cmul = [1 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100];
% Rval = [0.1 1 2 10 50 100];
% Qval = [0.1 1 2 10 50 100];
% val = {cmul,Rval,Qval};
% cellcomb = cell(1,numel(val));
% [cellcomb{:}] = ndgrid(val{:});
% cellcomb = cellfun(@(X) reshape(X,[],1),cellcomb,'UniformOutput',false);
% cellcomb = horzcat(cellcomb{:})
cellcomb = 1;
%%NETWORK STRUCTURES
% LINE STRUCTURE
Adj = tril(ones(6),-1) - tril(ones(6),-2);
Deg = diag(sum(Adj,2));
Pinning = zeros(6);
Pinning(1,1) = 1;
L = Deg- Adj;
c = 1/(2*min(real(eig(L+Pinning))));


%% Local Agent state dynamics control
Kctrl = acker(A,B,[p]);
A = A-B*Kctrl;

%Information gathering
umax = zeros(6,length(cellcomb));
ut0 = zeros(6,length(cellcomb));
d = zeros(2,length(cellcomb));
umaxn = zeros(6,length(cellcomb));
ut0n = zeros(6,length(cellcomb));
dn = zeros(2,length(cellcomb));
ctemp = c;


w=warning('off','all');
for i=1:length(cellcomb)
%Q R and c gridsearch
% c = ctemp*cellcomb(i,1);
% Q = eye(2)*cellcomb(i,2);
% R = cellcomb(i,3);
%Choose Q and R and cfrom test
Q = eye(2)*50;
R = 10;
c = ctemp*10;

%SVFB
%%Feedback GAIN control matrix K 
Pk = are(A,B*inv(R)*B',Q);
K = inv(R)*B'*Pk;

%% Observer gain Fn/Fl
Pn = are(A',C'*inv(R)*C,Q);
% Fn = Pneigh*C'*inv(R);
Fn = Pn*C'*inv(R);
%Local f
Fl = -Fn;

%Hurwitz check, if 
    if(Hurwitz(A,c,L,Pinning,Fn,C) == 0)
        disp("\t Variables were non Hurwitz\n")
        disp("Variable c:\n")
        disp(c)
        disp("Variable R:\n")
        disp(R)
        disp("Variable Q:\n")
        disp(Q)
        continue;
    end
    simOut = sim('neighbour_generalized');
    %%LOCAL INFOGATHER
    %Get time at with input is 0 for each agent
    ut0(1,i) = simOut.get('u1').time(find( abs(simOut.get('u1').data) >= -1e-3 ,1,'last'));
    ut0(2,i) = simOut.get('u2').time(find( abs(simOut.get('u2').data) >= -1e-3 ,1,'last'));
    ut0(3,i) = simOut.get('u3').time(find( abs(simOut.get('u3').data) >= -1e-3 ,1,'last'));
    ut0(4,i) = simOut.get('u4').time(find( abs(simOut.get('u4').data) >= -1e-3 ,1,'last'));
    ut0(5,i) = simOut.get('u5').time(find( abs(simOut.get('u5').data) >= -1e-3 ,1,'last'));
    ut0(6,i) = simOut.get('u6').time(find( abs(simOut.get('u6').data) >= -1e-3 ,1,'last'));
    
    %Max input for each agent
    umax(1,i) = max(abs(simOut.get('u1').data));
    umax(2,i) = max(abs(simOut.get('u2').data));
    umax(3,i) = max(abs(simOut.get('u3').data));
    umax(4,i) = max(abs(simOut.get('u4').data));
    umax(5,i) = max(abs(simOut.get('u5').data));
    umax(6,i) = max(abs(simOut.get('u6').data));
    
    %Delta is global disagreement
    delta6x1 =  simOut.get('delta6x1').data;
    delta6x2 =  simOut.get('delta6x2').data;
    
    delta6t =  simOut.get('delta6x1').time;
    %Find convergence time for each experiment 
    d(1,i) = delta6t( find( abs(delta6x1) >= 1e-3 ,1,'last'));
    d(2,i) = delta6t( find( abs(delta6x2) >= 1e-3 ,1,'last'));
    
    %%NEIGHBOURHOOD INFOGATHER
    ut0n(1,i) = simOut.get('un1').time(find( abs(simOut.get('un1').data) >= -1e-3 ,1,'last'));
    ut0n(2,i) = simOut.get('un2').time(find( abs(simOut.get('un2').data) >= -1e-3 ,1,'last'));
    ut0n(3,i) = simOut.get('un3').time(find( abs(simOut.get('un3').data) >= -1e-3 ,1,'last'));
    ut0n(4,i) = simOut.get('un4').time(find( abs(simOut.get('un4').data) >= -1e-3 ,1,'last'));
    ut0n(5,i) = simOut.get('un5').time(find( abs(simOut.get('un5').data) >= -1e-3 ,1,'last'));
    ut0n(6,i) = simOut.get('un6').time(find( abs(simOut.get('un6').data) >= -1e-3 ,1,'last'));
    
    
    umaxn(1,i) = max(abs(simOut.get('un1').data));
    umaxn(2,i) = max(abs(simOut.get('un2').data));
    umaxn(3,i) = max(abs(simOut.get('un3').data));
    umaxn(4,i) = max(abs(simOut.get('un4').data));
    umaxn(5,i) = max(abs(simOut.get('un5').data));
    umaxn(6,i) = max(abs(simOut.get('un6').data));

    delta6x1n =  simOut.get('delta6x1n').data;
    delta6x2n =  simOut.get('delta6x2n').data;
    delta6t =  simOut.get('delta6x1n').time;
    dn(1,i) = delta6t( find( abs(delta6x1n) >= 1e-3 ,1,'last'));
    dn(2,i) = delta6t( find( abs(delta6x2n) >= 1e-3 ,1,'last'));
end
% ind = find(cellcomb(:,1) == 1);
% [xx yy] = meshgrid(Rval,Qval)
% [a b] = meshgrid(dn(1,ind),dn(2,ind))
% mesh(xx,yy,reshape(dn(1,ind),[22 22]));
% plot(cellcomb(:,1),d(1,:))
% xlabel('c')
% ylabel('t')


function isHurwitz = Hurwitz(A,c,L,G,F,C)
isHurwitz = 1;
auto = eig(L+G);
    for i = 1:length(auto)

        if( max(real(eig(A-c*auto(i)*F*C))) > 0)
            isHurwitz = 0;
            break;
        end
    end
end
