%% Load 2019-01-24 batch first; sort by protocol
if ~exist('batch','var')
    load('D:\Data_Matlab\Batch_data\2019-01-24_batchdata_updated_struct_errorcorrect.mat')
end
[~,index] = sortrows({batch.policy_readable}.'); batch = batch(index); clear index

%% Preinitialize
Qn = zeros(45,1000); % Q(n)
QV_100_10 = zeros(45,1000); % DeltaQ_{100-10}(V)
QV_EOL_1 = zeros(45,1000); % DeltaQ_{EOL-1}(V)
QV_EOL = zeros(45,1000); % DeltaQ_{EOL}(V)

%% Loop
for k=1:45
    last_cycle = batch(k).cycle_life;
    
    Qn(k,1:(last_cycle+10)) = batch(k).summary.QDischarge(1:last_cycle+10);
    QV_100_10(k,:) = batch(k).cycles(100).Qdlin-batch(k).cycles(10).Qdlin;
    QV_EOL(k,:) = batch(k).cycles(last_cycle).Qdlin;
    QV_EOL_1(k,:) = batch(k).cycles(last_cycle).Qdlin - batch(k).cycles(1).Qdlin;
end

%% Save
csvwrite('Qn.csv',Qn)
csvwrite('QV_100_10.csv',QV_100_10)
csvwrite('QV_EOL.csv',QV_EOL)
csvwrite('QV_EOL_1.csv',QV_EOL_1)