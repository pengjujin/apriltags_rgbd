close all;
baseline_0_5 = [0.428, 0.203, 0.484, 0.457, 0.423, 0.441, 0.531, 0.107, 0.197, 0.508, 0.312, 0.201, 0.121];
test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
baseline_1_0 = [0.453, 0.381, 0.492, 0.452, 0.475, 0.463, 0.533, 0.292, 0.349, 0.468, 0.409, 0.453, 0.401]
baseline_0_2 = [0.304, 0.026, 0.395, 0.368, 0.352, 0.295, 0.533, 0, 0.014, 0.455, 0.115, 0.263, 0.107];
x = [30:5:90];
figure(1);
hold on;
ax = subplot(3, 1, 1);
scatter(x, baseline_0_2, 'MarkerFaceColor',[1 0 0]);
ax = subplot(3, 1, 2);
scatter(x, baseline_0_5, 'MarkerFaceColor', [0 1 0]);
ax = subplot(3, 1, 3);
scatter(x, baseline_1_0, 'MarkerFaceColor', [0 0 1]);

