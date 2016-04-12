classdef multiLayerPerceptron < handle
    %MULTILAYERPERCEPTRON for regression or binary classification
    % implementation of multi-layer perceptron (MLP),  also known as 
    % feed-forward neural network
    
    properties
        X       % input data
        y       % output data
        N       % number of data samples
        D       % data dimensionality
        W       % model weights
        grad    % current gradients
        a       % current hidden variable states
        z       % current non-linear hidden variable states
        layers  % number of network layers
        h       % activation function
        dh      % derivative of activation function
        mode    % output mode - classification or regression
    end
    
    methods
        function obj = multiLayerPerceptron(X, y, hiddens, activation, mode)
            % initialise data
            obj.X = X;
            obj.y = y;
            [obj.N, obj.D] = size(X);
            % initialise parameters
            hiddens = [hiddens, 1];
            obj.layers = length(hiddens);
            for i = 1 : obj.layers
                obj.a{i} = zeros(hiddens(i), 1);
                obj.z{i} = zeros(hiddens(i), 1);
            end
            network = [obj.D, hiddens];
            for i = 1 : length(network) - 1
                obj.W{i} = zeros(network(i + 1), network(i));
                obj.grad{i} = zeros(network(i + 1), network(i));
            end
            % initialise activation function and derivative
            if strcmp(activation, 'tanh')
                obj.h = @(x) tanh(x);
                obj.dh = @(x) 1 - tanh(x)^2;
            elseif strcmp(activation, 'sigma')
                obj.h = @(x) sigma(x);
                obj.dh = @(x) dSigma(x);
            else
                error('Invalid activation function');
            end
            if strcmp(mode, 'regression') || strcmp(mode, 'classification')
                obj.mode = mode;
            else
                error('Invalid mode');
            end
        end
        function forwardPropagate(obj, index)
            % update hidden variables with forward propagation
            obj.a{1} = obj.W{1} * obj.X(index, :)';
            obj.z{1} = obj.h(obj.a{1});
            % update interior layers
            for i = 2 : obj.layers
                obj.a{i} = obj.W{i} * obj.z{i - 1};
                obj.z{i} = obj.h(obj.a{i});
            end
        end
        function backPropagate(obj, index)
            % compute gradients with back propagation
            d = diag(obj.dh(obj.a{obj.layers}));
            % set leading delta
            if strcmp(obj.mode, 'regression')
                d = -d * 2 * (obj.y(index) - obj.z{obj.layers})';
            else % TODO classification
                d = d * 1;
            end
            for i = obj.layers : -1 : 2
                obj.grad{i} = d * obj.z{i - 1}';
                d = diag(obj.dh(obj.a{i})) * obj.W{i}' * d;
            end
            obj.grad{1} = d * obj.X(index, :);
        end
        function updateWeights(obj, alpha)
            % perform (stochastic) gradient descent step on weights
            for i = 1 : obj.layers
                obj.W{i} = gradientDescent(alpha, obj.W{i}, obj.grad{i});
            end
        end
        function train(obj, maxIters, alpha)
            % learn model weights
            iters = 0;
            while iters < maxIters
                cost = 0;
                % randomly permute data
                idx = randperm(obj.N);
                obj.X = obj.X(idx, :);
                obj.y = obj.y(idx, :);
                % run over epoch
                for i = 1 : obj.N
                    obj.forwardPropagate(i);
                    obj.backPropagate(i);
                    obj.updateWeights(alpha);
                    cost = cost + (obj.y(i) - obj.predict(obj.X(i)));
                end
                fprintf('Cost: %f\n', cost / obj.N);
                iters = iters + 1;
            end
        end
        function yhat = predict(obj, x)
            % make prediction by propagating forwards
            yhat = obj.h(obj.W{1} * x');
            for i = 2 : obj.layers
                yhat = obj.h(obj.W{i} * yhat);
            end
            if strcmp(obj.mode, 'classification')
                yhat = sigma(yhat);
            end
        end
    end
end
