% Implementation of naive Bayes classifier (NBC) for 3 Bernoulli features

% Generate data
N = 1000;

% Ground truth parameters
gPi = [0.3; 0.3; 0.4];
gTheta = [0.4, 0.1, 0.3 ; 0.2, 0.5, 0.9 ; 0.8, 0.3, 0.5];

y = mnrnd(1, gPi, N);
X = zeros(N, 3);

for i = 1:N
    X(i, 1) = binornd(1, gTheta(1, :) * y(i, :).');
    X(i, 2) = binornd(1, gTheta(2, :) * y(i, :).');
    X(i, 3) = binornd(1, gTheta(3, :) * y(i, :).');
end

% Split data
trY = y(1:(N/2),:);
teY = y((N/2 + 1):N,:);
trX = X(1:(N/2),:);
teX = X((N/2 + 1):N,:);

fprintf('Learning model parameters...\n');

% Learn model parameters
Pi = zeros(3, 1);
Theta = zeros(3, 3);

for i = 1:(N/2)
    c = find(trY(i,:), 1);
    Pi(c) = Pi(c) + 1;
    for j = 1:3
        if trX(i, j) == 1
            Theta(j, c) = Theta(j, c) + 1;
        end
    end
end

Theta(:, 1) = Theta(:, 1) ./ (Pi(1));
Theta(:, 2) = Theta(:, 2) ./ (Pi(2));
Theta(:, 3) = Theta(:, 3) ./ (Pi(3));
Pi = Pi ./ (N / 2);

predictions = zeros(N/2, 3);

fprintf('Predicting test data...\n');

% Predict data
for i=1:(N/2)
    p = zeros(3, 1);
    for c = 1:3
        p(c) = log(Pi(c));
        for j = 1:3
            if teX(i, j) == 1
                p(c) = p(c) + Theta(j, c);
            else
                p(c) = p(c) + (1 - Theta(j, c));
            end
        end
    end
    p = p ./ sum(p);
    [~, index] = max(p);
    predictions(i, index) = 1;
end

% Measure performance
numCorrect = 0;
for i=1:(N/2)
    if predictions(i, :) == teY(i, :)
        numCorrect = numCorrect + 1;
    end
end

fprintf('Success rate: %.2f (%d / %d)\n', 2.0 * numCorrect / N, numCorrect, N / 2);
