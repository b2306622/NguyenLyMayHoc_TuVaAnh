from processors import Data, Text, Image, Combined_Classifier, Search


class Ranker:
    def __init__(self,training_path, testing_path, n_classes):
        self.data = Data(training_path, testing_path, n_classes)

        #self.search = Search(self.data)
        #[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
        #self.search.do_combined_ranking(10,'euclidean', 0.7)
        #self.search.show_top(10, 10, 'text', 'q10_text.html')
        #self.search.show_top(10, 10, 'image', 'q10_image.html')
        #self.search.show_top(10, 10, 'combined', 'q10_combined.html')

        self.text = Text(self.data)
        self.text.do_test_clasifier(name='Naive Bayes')
        #self.text.do_test_clasifiers()
        #self.text.do_test_clustering()

        self.image = Image(self.data)
        self.image.do_test_clasifier(name='Naive Bayes')
        #self.image.do_test_clasifiers()
        #self.image.do_test_clustering()

        self.combined = Combined_Classifier(self.data, self.text, self.image)
        self.combined.do_combined_ranking(alpha=0.4)
        #self.combined.show_classification_result('classify_04.html')

