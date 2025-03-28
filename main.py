import argparse
import asyncio
import logging
import os
import time
import typeguard
from datetime import datetime
from typing import Literal
from litellm import RateLimitError  # Ensure litellm is imported
from forecasting_tools import *
from tenacity import retry, wait_exponential, stop_after_attempt

use_free_model = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress LiteLLM logging
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)
litellm_logger.propagate = False

# Custom logging handler to capture log messages
class RateLimitLogHandler(logging.Handler):
    global use_free_model
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            if "rate limit" in msg.lower() or "ratelimiterror" in msg.lower():  # Adjust if needed based on exact message
                use_free_model = False

        except Exception:
            self.handleError(record)

rate_limit_handler = RateLimitLogHandler()
rate_limit_handler.setLevel(logging.WARNING)
logger.addHandler(rate_limit_handler)
        
class Q1TemplateBot(ForecastBot):
    """
    This is a template bot that uses the forecasting-tools library to simplify bot making.

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    However generally the flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research for research_reports_per_question runs
        - Execute respective run_forecast function for `predictions_per_research_report * research_reports_per_question` runs
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```

    Check out https://github.com/Metaculus/forecasting-tools for a full set of features from this package.
    Most notably there is a built in benchmarker that integrates with ForecastBot objects.
    """

    _max_concurrent_questions = 1         # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    last_request_time = 0  # Track the last request time to enforce delay

    async def _research_and_make_predictions(self, *args, **kwargs):
        global use_free_model
        return await self._retry_on_rate_limit(super()._research_and_make_predictions, *args, **kwargs)
        # try:
        #     report = await self._retry_on_rate_limit(super()._research_and_make_predictions, *args, **kwargs)

        #     # Detect if it's a RateLimitError
        #     for error in report.errors:
        #         if "RateLimitError" in error:
        #             use_free_model = False
        #             logger.warning(f"RateLimitError detected: {error}")
        #             return RateLimitError("Rate limit exceeded")
        #     return report

        # except Exception as e:
        #     # Detect if it's a RateLimitError
        #     if "RateLimitError" in str(e):
        #         use_free_model = False
        #         logger.warning(f"RateLimitError detected: {e}")
        #     return e
        
    async def _run_individual_question(self, *args, **kwargs):
        global use_free_model
        return await self._retry_on_rate_limit(super()._run_individual_question, *args, **kwargs)
        # try:
        #     report = await self._retry_on_rate_limit(super()._run_individual_question, *args, **kwargs)

        #     # Detect if it's a RateLimitError
        #     for error in report.all_errors:
        #         if "RateLimitError" in error:
        #             use_free_model = False
        #             logger.warning(f"RateLimitError detected: {error}")
        #             return RateLimitError("Rate limit exceeded")
            
        #     return report

        # except Exception as e:
        #     # Detect if it's a RateLimitError
        #     if "RateLimitError" in str(e):
        #         use_free_model = False
        #         logger.warning(f"RateLimitError detected: {e}")
        #     return e

    async def _retry_on_rate_limit(self, func, *args, **kwargs):
        global use_free_model
        attempts = 0
        while attempts < 5:  # Maximum 5 retries
            try:
                report = await func(*args, **kwargs)
                for error in getattr(report, 'errors', []):
                    if "RateLimitError" in error:
                        raise RateLimitError("Rate limit exceeded")
                return report

            except Exception as e:
                if "RateLimitError" in str(e):
                    logger.info("hiiii found")
                    use_free_model = False
                    attempts += 1
                    wait_time = 2 ** attempts  # Exponential backoff (2, 4, 8, 16, 32 sec)
                    await asyncio.sleep(wait_time)
                    logger.warning(f"RateLimitError detected: {e}. Retrying in {wait_time} seconds...")
                else:
                    logger.error(f"Unexpected error: {e}")
                    return e
            
        logger.error("Max retries reached. Returning RateLimitError.")
        return RateLimitError("Rate limit exceeded after retries")

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(question.question_text, use_open_router=True)
            else:
                research = ""
            logger.info(f"Found Research for {question.page_url}:\n{research}")
            return research

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}

            Try to find base rates/historical rates and any way that the current situation is different from history.
            """
        )
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro" # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search.
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    def _get_final_decision_llm(self) -> GeneralLlm:
        global use_free_model
        logger.info(f"Running _get_final_decision_llm with use_free_model = {use_free_model}")
        # model = None
        # if os.getenv("OPENROUTER_API_KEY"):
        #     model = GeneralLlm(model="openrouter/openai/o3-mini-high", temperature=0.3)
        # else:
        #     raise ValueError("No API key for final_decision_llm found")
        # return model

        model = None
        try:
            logger.info(f"use_free_model = {use_free_model}")
            if use_free_model and os.getenv("OPENROUTER_API_KEY"):
                logger.info("Attempting to use free model")                    
                # Enforce a 5-second delay between requests
                time_since_last_request = time.time() - self.last_request_time
                if time_since_last_request < 6:
                    time.sleep(6 - time_since_last_request)
                self.last_request_time = time.time()  # Update request time
                
                model = GeneralLlm(model="openrouter/deepseek/deepseek-r1:free", temperature=0.3)

            elif os.getenv("OPENROUTER_API_KEY") and not use_free_model:
                logger.info("Attempting to use paid model")    
                model = GeneralLlm(model="openrouter/deepseek/deepseek-r1", temperature=0.3)

            else:
                raise ValueError("No API key for final_decision_llm found")

            return model  # Return model if no exceptions occur

        except Exception as e: # it's comming from a warning and in the message
            logger.info(f"Error: {str(e)}")
            raise  # Raise other errors immediately
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            (e) Please consider the historical base rate and make a guess if you are not sure.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message

def summarize_reports(forecast_reports: list[ForecastReport | BaseException]) -> None:
    valid_reports = [
        report for report in forecast_reports if isinstance(report, ForecastReport)
    ]
    exceptions = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]
    minor_exceptions = [
        report.errors for report in valid_reports if report.errors
    ]

    for report in valid_reports:
        question_summary = clean_indents(f"""
            URL: {report.question.page_url}
            Errors: {report.errors}
            Summary:
            {report.summary}
            ---------------------------------------------------------
        """)
        logger.info(question_summary)

    if exceptions:
        # raise RuntimeError(
        #     f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        # )
        logger.error(
            f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        )
    if minor_exceptions:
        logger.error(
            f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    # make changes, change last back to true
    template_bot = Q1TemplateBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True, 
    )

    if run_mode == "tournament":
        Q4_2024_AI_BENCHMARKING_ID = 32506
        Q1_2025_AI_BENCHMARKING_ID = 32627
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_AI_BENCHMARKING_ID, return_exceptions=True
            )
        )
            
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        Q1_2025_QUARTERLY_CUP_ID = 32630
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(
                questions, return_exceptions=True
            )
        )
    forecast_reports = typeguard.check_type(forecast_reports, list[ForecastReport | BaseException])
    summarize_reports(forecast_reports)
