classdef multiLayerPerceptron
    %MULTILAYERPERCEPTRON Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X, y, N, D, W, grad, a, z, layers, activation, mode
    end
    
    methods
        function obj = multiLayerPerceptron(X, y, hiddens, activation, mode)
            % initialise data
            obj.X = X;
            obj.y = y;
            [obj.N, obj.D] = size(X);
            % initialise parameters
            % TODO: Assume scalar output for now
            hiddens = [hiddens, 1];
            obj.layers = length(hiddens);
            for i = 1 : obj.layers
                obj.a{i} = zeros(hiddens(i), 1);
                obj.z{i} = zeros(hiddens(i), 1);
            end
            % TODO: Assume scalar output for now
            network = [obj.D, hiddens];
            for i = 1 : length(network) - 1
                obj.W{i} = zeros(network(i), network(i + 1));
                obj.grad{i} = zeros(network(i), network(i + 1));
            end
            % initialise activation function
            if strcmp(activation, 'tanh') || strcmp(activation, 'sigma')
                obj.activation = activation;
            else
                throw(exception);
            end
            % initialise mode
            if strcmp(mode, 'regression') || strcmp(mode, 'classification')
                obj.mode = mode;
            else
                throw(exception);
            end
        end
        function forwardPropagate(obj, index)
            % update hidden variables with forward propagation
            prop = obj.X(index, :);
            for i = 1 : obj.layers
                prop = prop * obj.W{i};
                obj.a{i} = prop;
                % apply activation function
                if strcmp(obj.activation, 'tanh')
                    prop = tanh(prop);
                else
                    prop = sigma(prop);
                end
                obj.z{i} = prop;
            end
        end
        function delta = deltaStep(obj, layer)
            % returns delta step for back propagation
            if strcmp(obj.activation, 'tanh')
                delta = diag(sech(obj.a{layer})^2);
            else
                delta = diag(dSigma(obj.a{layer}));
            end
        end
        function backPropagate(obj, index)
            % update weights with back propagation
            delta = obj.deltaStep(obj.layers);
            % set leading delta
            if strcmp(obj.mode, 'regression')
                delta = -delta * 2 * (obj.y(index) - obj.z{obj.layers});
            else % TODO classification
            end
            for i = obj.layers : -1 : 2
                obj.grad{i} = delta * obj.z{i - 1};
                delta = obj.deltaStep(i) * obj.W{i} * delta;
            end
            obj.grad{1} = delta * obj.X(index, :);
        end
        function updateWeights(obj)
            % perform (stochastic) gradient descent step on weights
            for i = 1 : obj.layers
                obj.W{i} = gradientDescent(0.01, obj.W{i}, obj.grad{i});
            end
        end
        function train(obj, maxIters)
            % learn model weights
            iters = 0;
            while iters < maxIters
                % run over epoch
                for i = 1 : obj.N
                    obj.forwardPropagate(i);
                    obj.backPropagate(i);
                    obj.updateWeights();
                end
                iters = iters + 1;
            end
        end
%         function predict(obj, x)
%             % same as forward propagate
%         end
    end
end
