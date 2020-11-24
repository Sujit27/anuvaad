import C from "../constants";
import API from "./api";
import ENDPOINTS from "../../../configs/apiendpoints";

export default class FetchQuestions extends API {
  constructor(timeout = 2000) {
    super("GET", timeout, false);
    this.type = C.FEEDBACK_QUESTIONS;

    this.endpoint = `${super.apiEndPointAuto()}${ENDPOINTS.fetchfeedbackpending}`;
    this.feedbackQuestions = [];
  }

  toString() {
    return `${super.toString()} , type: ${this.type}`;
  }

  processResponse(res) {
    super.processResponse(res);
    if (res.data) {
      this.feedbackQuestions = res.data;
    }
  }

  apiEndPoint() {
    return this.endpoint;
  }

  getBody() {
    return {};
  }

  getHeaders() {
    return {
      headers: {
         'auth-token': `${decodeURI(localStorage.getItem("token"))}`,
        "Content-Type": "application/json"
      }
    };
  }

  getPayload() {
    return this.feedbackQuestions;
  }
}
