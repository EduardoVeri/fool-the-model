import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Classe wrapper para o classificador XGBoost utilizando xgboost.XGBClassifier
class XGBClassifierWrapper:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
                 use_label_encoder=False, eval_metric='logloss'):
        """
        Inicializa o modelo XGBoost com os parâmetros básicos.
        
        Parâmetros:
          - n_estimators: Número de árvores a serem construídas.
          - learning_rate: Taxa de aprendizado.
          - max_depth: Profundidade máxima de cada árvore.
          - random_state: Semente para reprodutibilidade.
          - use_label_encoder: Desativa o uso do label encoder interno (recomendado para versões mais recentes).
          - eval_metric: Métrica de avaliação utilizada durante o treinamento.
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric
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
