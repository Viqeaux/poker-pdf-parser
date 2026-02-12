{
  "openapi": "3.1.0",
  "servers": [{ "url": "https://poker-pdf-parser.onrender.com" }],
  "info": { "title": "Poker PDF Parser", "version": "0.3.0" },
  "paths": {
    "/parse_poker_pdf": {
      "post": {
        "summary": "Parse Poker PDF",
        "operationId": "parse_poker_pdf",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/ParseRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/ParseResponse" }
              }
            }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "OpenAIFileRef": {
        "description": "Either an object containing download_link (preferred) or a string (URL or mistaken file_...).",
        "anyOf": [
          {
            "type": "object",
            "properties": {
              "download_link": { "type": "string" }
            },
            "required": ["download_link"],
            "additionalProperties": true
          },
          { "type": "string" }
        ]
      },
      "ParseRequest": {
        "type": "object",
        "required": ["openaiFileIdRefs"],
        "properties": {
          "openaiFileIdRefs": {
            "type": "array",
            "items": { "$ref": "#/components/schemas/OpenAIFileRef" }
          },
          "options": {
            "type": ["object", "null"],
            "default": null,
            "additionalProperties": true
          }
        },
        "additionalProperties": false
      },
      "ParseResponse": {
        "type": "object",
        "required": ["hand_history_text", "hands_detected", "BUILD_ID", "ocr_available", "errors_count"],
        "properties": {
          "BUILD_ID": { "type": "string" },
          "ocr_available": { "type": "boolean" },
          "errors_count": { "type": "integer" },
          "hands_detected": { "type": "integer" },
          "hand_history_text": { "type": "string" }
        },
        "additionalProperties": true
      },
      "HTTPValidationError": {
        "type": "object",
        "properties": {
          "detail": {
            "type": "array",
            "items": { "$ref": "#/components/schemas/ValidationError" }
          }
        }
      },
      "ValidationError": {
        "type": "object",
        "required": ["loc", "msg", "type"],
        "properties": {
          "loc": {
            "type": "array",
            "items": { "anyOf": [{ "type": "string" }, { "type": "integer" }] }
          },
          "msg": { "type": "string" },
          "type": { "type": "string" }
        }
      }
    }
  }
}
