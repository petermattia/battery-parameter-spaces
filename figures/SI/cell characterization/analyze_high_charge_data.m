% Peter Attia
% February 5, 2018
% This script extracts and visualizes temperature distributions for
%   high-discharge rate tests on A123 LFP/graphite cells.

clear, close all
filename_saving = 'ch11';

if strcmp(filename_saving, 'ch9')
    filename_loading = '2018-12-05-high_charge_rate_test_CH9.csv';
elseif strcmp(filename_saving, 'ch10')
    filename_loading = '2018-12-05-high_charge_rate_test_CH10.csv';
elseif strcmp(filename_saving, 'ch11')
    filename_loading = '2018-12-05-high_charge_rate_test_CH11.csv';
end

%% Load xlsx file
if ~exist('data','var')
    data = csvread(filename_loading,1,0);
end

%% Extract columns of interest
t = data(:,2)./60; % test time, seconds -> minutes
step_idx = data(:, 5); % step index
V = data(:,8); % voltage
I = data(:,7); % current
Qc = data(:,9); % charge capacity
Qd = data(:,10); % discharge capacity
T = data(:,15); % surface temperature

%% Pre-initialize arrays/cell arrays
step_idx_list = 2:4:40;
n_cycles = length(step_idx_list);
t_cycle = cell(n_cycles,1);
V_cycle = cell(n_cycles,1);
I_cycle = cell(n_cycles,1);
Qc_cycle = cell(n_cycles,1);
T_cycle = cell(n_cycles,1);
I_leg = cell(n_cycles,1); % legend with rates

%% Extract charge cycles
for k = 1:n_cycles
    step_indices = find(step_idx == step_idx_list(k));
    t_cycle{k} = t(step_indices) - t(step_indices(1));
    V_cycle{k} = V(step_indices);
    I_cycle{k} = I(step_indices);
    Qc_cycle{k} = Qc(step_indices) - Qc(step_indices(1));
    T_cycle{k} = T(step_indices);
end

%% Extract I vs step
for k = 1:n_cycles
    I_leg{k} = strcat(num2str(round(mean(I_cycle{k})/1.1)),'C'); % find discharge C rate
end

%% Save data
mkdir(filename_saving)
cd(filename_saving)

save(strcat(filename_saving,'.mat'),'V_cycle', 'I_cycle', 'Qc_cycle', 'T_cycle', 'I_leg')

%% Initialize colors
pink = [255, 192, 203]/255;
red = [1, 0, 0];
colors_p = [linspace(pink(1),red(1),n_cycles)', linspace(pink(2),red(2),n_cycles)', linspace(pink(3),red(3),n_cycles)'];

light_blue = [203, 192, 255]/255;
blue = [0, 0, 1];
colors_b = [linspace(light_blue(1),blue(1),n_cycles)', linspace(light_blue(2),blue(2),n_cycles)', linspace(light_blue(3),blue(3),n_cycles)'];


%% Plot V vs Qd
for k = 1:n_cycles
    plot(Qc_cycle{k}, V_cycle{k}, 'color', colors_p(k,:)), hold on
end
legend(I_leg,'location','southeast'), legend boxoff
xlabel('Capacity (Ah)'), ylabel('Voltage (V)')
print('VvsQ','-dpng')

%% Plot T vs Qd
figure
for k = 1:n_cycles
    plot(Qc_cycle{k}, T_cycle{k}, 'color', colors_b(k,:)), hold on
end
legend(I_leg, 'location','northeast'), legend boxoff
xlabel('Capacity (Ah)'), ylabel('Temperature (\circC)')
print('TvsQ','-dpng')

%% Plot T vs t
figure
for k = 1:n_cycles
    plot(t_cycle{k}, T_cycle{k}, 'color', colors_b(k,:)), hold on
end
legend(I_leg, 'location','northeast'), legend boxoff
xlabel('Time (min)'), ylabel('Temperature (\circC)')
print('Tvst','-dpng')

cd ..