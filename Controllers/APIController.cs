using Microsoft.AspNetCore.Components.Forms;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;
using System.Diagnostics;
using static Email_Classifier_API_v1.MLModel;

namespace Email_Classifier_API_v1.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class APIController : ControllerBase
    {
        [HttpPost]
        [Route("Predict")]
        public ModelOutput Predict(InputText Input)
        {
            ModelInput input = new()
            {
                Text = Input.Text.ToLower(),
            };
            Debug.WriteLine(Input.Text);
            var prediction = MLModel.Predict(input);
            return prediction;
        }
        public class InputText
        {
            public string Text { get; set; } = string.Empty;
        }
    }
}
