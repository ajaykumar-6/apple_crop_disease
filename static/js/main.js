$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    let isPredicting = false; // 🔒 Flag to prevent double-click

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.image-section').fadeIn(500);
                $('#btn-predict').show().prop('disabled', false);
                $('#result').hide();
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        readURL(this);
    });

    $('#btn-predict').click(function () {
        if (isPredicting) return; // 🚫 Prevent multiple clicks
        isPredicting = true;

        var form_data = new FormData($('#upload-file')[0]);

        // UI feedback
        $(this).prop('disabled', true).text('Predicting...');
        $('.loader').show();
        $('#result').fadeOut(300);

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            timeout: 30000, // optional: prevent hanging requests
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600).html(data);
                $('#btn-predict').prop('disabled', false).text('🔍 Predict Again');
                isPredicting = false;
            },
            error: function (xhr, status, error) {
                $('.loader').hide();
                $('#result').fadeIn(600).html(
                    `<div class='alert alert-danger'>
                        <strong>Error:</strong> Unable to process image. Please try again.
                    </div>`
                );
                $('#btn-predict').prop('disabled', false).text('🔍 Try Again');
                isPredicting = false;
            }
        });
    });
});
