from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Classe wrapper para a Decision Tree utilizando scikit-learn
class DTClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Inicializa o modelo de Decision Tree com os parâmetros básicos.
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    
    def fit(self, X_train, y_train):
        """
        Treina o modelo utilizando os dados de treinamento.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Realiza previsões para os dados fornecidos.
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Retorna a acurácia do modelo nos dados fornecidos.
        """
        return self.model.score(X, y)
    
    def predict_proba(self, X):
        """
        Retorna as probabilidades preditas para cada classe.
        """
        return self.model.predict_proba(X)
    
    def tune_parameters(self, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
        """
        Otimiza os hiperparâmetros utilizando GridSearchCV.
        
        Parameters:
          - param_grid: Dicionário com os hiperparâmetros a serem testados.
          - X_train, y_train: Dados de treinamento.
          - cv: Número de folds para validação cruzada.
          - scoring: Métrica utilizada para avaliação.
          
        Retorna:
          - best_params: Melhores parâmetros encontrados.
          - best_score: Melhor pontuação obtida.
        """
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        # Atualiza o modelo com o melhor estimador encontrado
        self.model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
