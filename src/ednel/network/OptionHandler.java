package ednel.network;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;

/**
 * Handles options of variables.
 */
public class OptionHandler {
    private JSONObject options;

    public OptionHandler() throws IOException, ParseException {
        JSONParser jsonParser = new JSONParser();

        InputStream stream_options = this.getClass().getClassLoader().getResourceAsStream("options.json");
        this.options = (JSONObject)jsonParser.parse(
                new BufferedReader(
                        new InputStreamReader(
                                stream_options,
                                StandardCharsets.UTF_8)
                )
        );
    }

    public HashMap<String, String> handle(HashMap<String, String> optionTable, String variableName, String algorithmName, String sampledValue) throws ParseException {
        JSONObject optionObj = (JSONObject)options.getOrDefault(variableName, null);
        if(optionObj == null) {
            String algorithmOptions = optionTable.getOrDefault(algorithmName, "");

            // if algorithm options is expecting a sampled value
            // from this variable, replace it in the options string
            if(algorithmOptions.contains(variableName)) {
                optionTable.put(algorithmName, algorithmOptions.replace(variableName, sampledValue));
            }

            optionObj = (JSONObject)options.getOrDefault(sampledValue, null);
        }

        // checks whether this is an option
        if(optionObj != null) {
            Boolean presenceMeans = (Boolean)optionObj.get("presenceMeans");
            String optionName = String.valueOf(optionObj.get("optionName"));
            String dtype = String.valueOf(optionObj.get("dtype"));

            if(dtype.equals("np.bool")) {
                if(String.valueOf(sampledValue).toLowerCase().equals("false")) {
                    if(!presenceMeans) {
                        optionTable.put(algorithmName, (optionTable.getOrDefault(algorithmName, "") + " " + optionName).trim());
                    }
                }
                if(String.valueOf(sampledValue).toLowerCase().equals("true")) {
                    if(presenceMeans) {
                        optionTable.put(algorithmName, (optionTable.getOrDefault(algorithmName, "") + " " + optionName).trim());
                    }
                }
            } else if(dtype.equals("dict")) {
                JSONObject dict = (JSONObject)((new JSONParser()).parse(optionName));

                optionTable.put(
                        algorithmName, (
                                optionTable.getOrDefault(algorithmName, "") + " " + dict.get(sampledValue)
                        ).trim()
                );
            } else {
                optionTable.put(
                        algorithmName, (
                                optionTable.getOrDefault(algorithmName, "") + " " + optionName + " " + sampledValue
                        ).trim()
                );
            }
        }
        return optionTable;
    }

}
