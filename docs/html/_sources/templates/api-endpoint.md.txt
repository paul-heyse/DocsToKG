# 1. API Endpoint: <ENDPOINT_PATH>

**Method**: <HTTP_METHOD>

**Description**: [Brief description of what this endpoint does within DocsToKG]

**Authentication**: [Required/Opcional authentication method]

## 2. Parameters

### Path Parameters

- `param_name` (type): Description of the parameter

### Query Parameters

- `param_name` (type, optional): Description of the parameter

### Request Body

```json
{
  "field_name": "field_type",
  "optional_field": "field_type"
}
```

### Headers

- `Header-Name`: Description of header requirement

## 3. Responses

### Success Response

**Status Code**: `200`

**Response Body**:

```json
{
  "field_name": "response_type",
  "status": "success"
}
```

### Error Responses

- **Status Code**: `400` - Bad Request
  - **Description**: [When this error occurs]
  - **Response Body**:

    ```json
    {
      "error": "error_message",
      "code": "ERROR_CODE"
    }
    ```

- **Status Code**: `401` - Unauthorized
  - **Description**: [When this error occurs]

- **Status Code**: `404` - Not Found
  - **Description**: [When this error occurs]

- **Status Code**: `500` - Internal Server Error
  - **Description**: [When this error occurs]

## 4. Examples

### cURL Example

```bash
curl -X <METHOD> <FULL_URL> \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'
```

### Python Example

```python
import requests

response = requests.METHOD(
    "<FULL_URL>",
    json={"key": "value"},
    headers={"Authorization": "Bearer <TOKEN>"}
)
print(response.json())
```

### JavaScript Example

```javascript
const response = await fetch("<FULL_URL>", {
  method: "<METHOD>",
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer <TOKEN>'
  },
  body: JSON.stringify({key: 'value'})
});

const data = await response.json();
console.log(data);
```

## 5. Related Endpoints

- Add links to closely related endpoints here (if applicable)

## 6. Notes

- [Additional implementation notes]
- [Rate limiting information]
- [Version information]
