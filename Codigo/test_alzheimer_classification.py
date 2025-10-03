from alzheimer_classifier import ImageClassifier
import os

def main():
    classifier = ImageClassifier()
    classifier.set_images_dir("./OriginalDataset")
    
    # parâmetros do modelo
    classifier.set_n_colors(32)
    classifier.set_gaussian_radius(0)
    classifier.set_sharpness_boost_strength(1.4)
    classifier.set_contrast_boost_strength(2.2)
    classifier.set_brightness_boost_strength(1.1)
    classifier.set_percentage_train(75)  # 75% para treino
    
    # configurações de segmentação
    classifier.set_min_object_size(100)
    classifier.set_gaussian_sigma(1.0)
    
    print("=== Classificador de Níveis de Alzheimer com Segmentação Otsu ===")
    print("Configurações:")
    print(f"- Diretório: {classifier.get_images_dir()}")
    print(f"- Número de cores: {classifier.get_n_colors()}")
    print(f"- Porcentagem para treino: {classifier.get_percentage_train()}%")
    print(f"- Segmentação: Habilitada (Threshold de Otsu)")
    print(f"- Tamanho mínimo do objeto: {classifier.get_min_object_size()}")
    print(f"- Sigma gaussiano: {classifier.get_gaussian_sigma()}")
    print(f"- Classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented")
    
    try:
        print("Iniciando separação dos dados de treino e teste...")
        classifier.split_train_test()
        print("[OK] Separação concluída!")
        
        print("Iniciando treinamento do modelo...")
        classifier.train_model()
        print("[OK] Treinamento concluído!")
        
        print("Fazendo previsões nas imagens de teste...")
        classifier.predict_test_images()
        print("[OK] Previsões concluídas!")
        
        print("Calculando métricas de avaliação...")
        accuracy, specificity = classifier.get_prediction_metrics()
        print("[OK] Métricas calculadas!")
        
        print("\n=== RESULTADOS ===")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Especificidade: {specificity:.4f}")
        
        # métricas de tempo
        print("\n=== MÉTRICAS DE TEMPO ===")
        for metric, time in classifier.get_runtime_metrics().items():
            print(f"{metric}: {time}")
        
        print("\nArquivos gerados:")
        print("- matriz-confusao.png")
        
        print("\n[OK] Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"ERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()
        print("Verifique se o dataset está no formato correto e se todas as dependências estão instaladas.")

if __name__ == "__main__":
    main()
