using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Email_Classifier_API_v1
{
    public partial class MLModel
    {
        
        public static List<Tuple<string, double>> CalculatePFI(MLContext mlContext, IDataView trainData, ITransformer model, string labelColumnName)
        {
            var preprocessedTrainData = model.Transform(trainData);

            var permutationFeatureImportance =
         mlContext.MulticlassClassification
         .PermutationFeatureImportance(
                 model,
                 preprocessedTrainData,
                 labelColumnName: labelColumnName);

            var featureImportanceMetrics =
                 permutationFeatureImportance
                 .Select((kvp) => new { kvp.Key, kvp.Value.MacroAccuracy })
                 .OrderByDescending(myFeatures => Math.Abs(myFeatures.MacroAccuracy.Mean));

            var featurePFI = new List<Tuple<string, double>>();
            foreach (var feature in featureImportanceMetrics)
            {
                var pfiValue = Math.Abs(feature.MacroAccuracy.Mean);
                featurePFI.Add(new Tuple<string, double>(feature.Key, pfiValue));
            }

            return featurePFI;
        }
    }
}


