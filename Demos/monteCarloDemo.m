N = 2500;

x = -1 + 2 * rand(N, 1);
y = -1 + 2 * rand(N, 1);

xin = []; yin = []; xout = []; yout = [];

for i = 1:N
    if (x(i)^2 + y(i)^2) < 1
        xin = [xin,x(i)];
        yin = [yin,y(i)];
    else
        xout = [xout,x(i)];
        yout = [yout,y(i)];
    end
end

pi = 4 * length(xin)/ N;

plot(xin, yin, 'o', 'Color', 'red', 'MarkerSize', 7, 'MarkerFaceColor', 'red');
hold on;
plot(xout, yout, 'o', 'Color', 'blue', 'MarkerSize', 7, 'MarkerFaceColor', 'blue');
grid on;