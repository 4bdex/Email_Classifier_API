
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace Email_Classifier_API_v1
{
    public partial class MLModel
    {
        public const string RetrainFilePath = @"D:\Abdex\Desktop\sentiment labelled sentences\yelp_labelled.txt";
        public const char RetrainSeparatorChar = '	';
        public const bool RetrainHasHeader = true;

     
        public static void Train(string outputModelPath, string inputDataFilePath = RetrainFilePath, char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader)
        {
            var mlContext = new MLContext();

            var data = LoadIDataViewFromFile(mlContext, inputDataFilePath, separatorChar, hasHeader);
            var model = RetrainModel(mlContext, data);
            SaveModel(mlContext, model, data, outputModelPath);
        }

        public static IDataView LoadIDataViewFromFile(MLContext mlContext, string inputDataFilePath, char separatorChar, bool hasHeader)
        {
            return mlContext.Data.LoadFromTextFile<ModelInput>(inputDataFilePath, separatorChar, hasHeader);
        }



 
        public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
        {
            DataViewSchema dataViewSchema = data.Schema;

            using (var fs = File.Create(modelSavePath))
            {
                mlContext.Model.Save(model, dataViewSchema, fs);
            }
        }


   
        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"text", outputColumnName: @"text")
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"text" }))
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"label", inputColumnName: @"label", addKeyValueAnnotationsAsText: false))
                                    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options() {LabelColumnName = @"label", FeatureColumnName = @"Features" }))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

            return pipeline;
        }
    }
}
