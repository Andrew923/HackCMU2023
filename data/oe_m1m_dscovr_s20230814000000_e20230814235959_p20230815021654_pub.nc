CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230814000000_e20230814235959_p20230815021654_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-15T02:16:54.967Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-14T00:00:00.000Z   time_coverage_end         2023-08-14T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�z@  
�          @�(������k��}p��^=qCv�H�����r�\��p���z�Cw�                                     Bx���  "          @����z��p  �h���LQ�Cyn��z��vff��\)�vffCy��                                    Bx���  �          @}p�����XQ쿚�H��=qCuz����a녿!G��G�Cvk�                                    Bx��2  �          @l�Ϳ�\)�(��33���ChxR��\)�/\)��\)��Ck�                                    Bx���  "          @g
=����#�
��Q���  Ci�����333���R���Cl�                                    Bx��~  "          @a녿�{�Q��p����CpG���{�-p������Csh�                                    Bx��$  �          @e���
=�.�R��33�\)Cv����
=�@�׿����
Cx��                                    Bx���  �          @mp�����<�Ϳ�Q��ڣ�Ct�3����L(�������CvaH                                    Bx��p  �          @u�����n{?��A{C�p�����dz�?�A���C�B�                                    Bx��  T          @u���\)�h��?��A�\)C����\)�Y��?޸RA�  C��f                                    Bx��  
�          @u��!G��qG�>W
=@K�C�Ff�!G��k�?W
=AJ�RC�*=                                    Bx�b  
�          @r�\���mp�>���@�Q�C������g
=?h��A`��C��                                    Bx�*  �          @qG��W
=�hQ�?p��Ah��C�T{�W
=�Z�H?��A�p�C�9�                                    Bx�8�  �          @mp���\)�^{?�p�A��C�h���\)�N{?��A��HC�\)                                    Bx�GT  
�          @l(�>W
=�Z�H?���A��HC���>W
=�I��?��A�(�C��\                                    Bx�U�  �          @h��>���`��?c�
Ab�\C�AH>���S�
?��HA���C�U�                                    Bx�d�  �          @g����
�]p�?O\)AS�C��ͼ��
�Q�?��A���C��=                                    Bx�sF  "          @g��#�
�W�?�ffA���C����#�
�Fff?���A��RC���                                    Bx���  "          @i�����
�P��?�33A�33C��ü��
�;�@�B��C���                                    Bx���  �          @j=q=�\)�S�
?ǮA�p�C���=�\)�@  @BC��                                     Bx��8  �          @o\)���Z=q?�33A��RC�:���HQ�?��HA�C���                                    Bx���  T          @u����b�\?�z�A�
=C�1����P  @   A���C���                                    Bx���  �          @u���z��c33?�33A��C��f��z��P��?��RA�C�p�                                    Bx��*  �          @j�H�u�W�?�
=A�\)C�녾u�Dz�?�p�B�HC���                                    Bx���  �          @p  ��\)�`  ?��A�C�����\)�N�R?�\)A�=qC��                                    Bx��v  �          @s33��33�fff?��A��C�9���33�Vff?޸RA�ffC�                                    Bx��  �          @q녾�33�hQ�?�  Aup�C�:ᾳ33�Y��?�{A�\)C��                                    Bx��  �          @s�
�u�hQ�?��A�33C�q�u�XQ�?�  A�(�C���                                    Bx�h  �          @y����  �k�?�G�A��HC�{��  �Z=q?�\)A��C��                                    Bx�#  "          @|(��8Q��j=q?��HA��C����8Q��Vff@�
A���C�xR                                    Bx�1�  
�          @y���#�
�P  @   A�z�C��\�#�
�6ff@!�B&  C���                                    Bx�@Z  T          @u>.{�:�H@�RB!Q�C���>.{���@<��BI�C��                                    Bx�O   �          @s33=����>{@
=B�RC��=����!G�@5�BA��C��                                    Bx�]�  T          @u<��7
=@#�
B'ffC�H�<���@AG�BOffC�XR                                    Bx�lL  �          @s�
�L���>{@��BQ�C����L��� ��@7�BCffC�u�                                    Bx�z�  �          @n�R<��G
=@�\B�HC�C�<��-p�@#33B-  C�N                                    Bx���  T          @c�
>���\)@!G�B4�RC��\>��� ��@:=qB\��C��H                                    Bx��>  �          @^�R?���,(�@
=BQ�C��f?���G�@#33B=�C�Ǯ                                    Bx���  �          @n�R>��H��@>�RBW�RC��{>��H�Ǯ@S�
B~��C�˅                                    Bx���  �          @s33>u��\@L(�Be(�C�ff>u��Q�@`  B��{C��                                    Bx��0  �          @w
=>�Q�� ��@QG�Bh33C�3>�Q쿳33@e�B��fC�1�                                    Bx���  �          @~{?(����@h��B��C�,�?(��.{@tz�B�#�C��                                     Bx��|  �          @xQ�?   ��Q�@g�B�ǮC�p�?   ��@q�B�{C��\                                    Bx��"  T          @q�>�
=��z�@^{B���C�^�>�
=�E�@j=qB��HC�Y�                                    Bx���  T          @U=�Q���@{BA�C�4{=�Q���H@4z�Bj�C���                                    Bx�n  T          @333��33�   �#�
����Cp���33���R>B�\@��\Cp\                                    Bx�  
�          @C�
�Q��ff��z����Cx�)�Q���=��
@Q�CxǮ                                    Bx�*�  �          @\(�>B�\�>{?�A�C�Ǯ>B�\�(Q�@
�HB�
C��                                    Bx�9`  "          @Z=q>\)�6ff?ٙ�A��RC�j=>\)�   @(�B$ffC��)                                    Bx�H  �          @l�ͽL���>{@��B  C��=�L���"�\@(Q�B833C�w
                                    Bx�V�  �          @j�H>k��z�@>{B[C�'�>k���G�@S33B�W
C�S3                                    Bx�eR  
Z          @j�H>�G��\)@7�BN33C���>�G���
=@N�RBv�C�P�                                    Bx�s�  �          @qG�>���@A�BV\)C�@ >�����@W�B~z�C�g�                                    Bx���  �          @n{>�33��@5BGz�C�9�>�33����@N{Bpp�C�z�                                    Bx��D  �          @l��=�\)���@:�HBP�RC���=�\)�ٙ�@Q�BzQ�C�AH                                    Bx���  �          @p  �L�����@?\)BSC�c׽L�Ϳ�
=@VffB}p�C�.                                    Bx���  �          @{�>aG���@VffBjQ�C�&f>aG����@j=qB��)C��{                                    Bx��6  �          @o\)�u�   @HQ�Bd��C�� �u��33@\(�B�8RC�0�                                    Bx���  �          @s�
=�G����@c�
B��RC�` =�G��+�@p  B��=C��)                                    Bx�ڂ  �          @s33>��
�Q�@l(�B��C��q>��
��@q�B�k�C���                                    Bx��(  �          @l(�>�p��\)@j=qB��\C��=>�p�?�@hQ�B�B[ff                                    Bx���  
�          @vff?   �.{@s�
B�#�C���?   ?�@q�B�#�B9�                                    Bx�t  
Z          @xQ�>�<�@vffB�Q�@HQ�>�?:�H@q�B�=qBg{                                    Bx�  
�          @tz�?#�
>\)@o\)B�.AJ{?#�
?Tz�@i��B�
=BQ��                                    Bx�#�  �          @o\)?z�=���@k�B��)A��?z�?G�@fffB��)BU(�                                    Bx�2f  �          @z�H?�G����
@p��B�k�C�(�?�G�>\@p��B��HA�G�                                    Bx�A  �          @~{?�z�\@qG�B�{C��q?�z�>���@qG�B��=A{
=                                    Bx�O�  
�          @w
=?c�
=�@l��B��q@�{?c�
?O\)@g�B�=qB)(�                                    Bx�^X  
�          @u�?(�����@p  B�aHC�4{?(�?��@mp�B�#�B2(�                                    Bx�l�  "          @���?333���@|(�B�\)C��f?333?
=@y��B���B=q                                    Bx�{�  T          @���?L�Ϳ:�H@��B���C��
?L��=u@�
=B�u�@��                                    Bx��J  �          @�\)?c�
�J=q@�=qB�(�C�8R?c�
<�@�z�B�@                                       Bx���  �          @�33?W
=��=q@��
B�\)C��{?W
=��z�@�  B�u�C��q                                    Bx���  "          @��H?u��ff@�33B��3C�8R?u���@�
=B�z�C�`                                     Bx��<  �          @�
=?.{�p��@��B�{C��?.{�#�
@��B��C�ff                                    Bx���  "          @�
=?.{�fff@��B��fC�b�?.{��@���B�8RC�ٚ                                    Bx�ӈ  
�          @�G�?E��O\)@xQ�B��HC���?E��u@~{B���C���                                    Bx��.  �          @��?@  �Q�@��HB�u�C�8R?@  ��@�p�B�� C���                                    Bx���  "          @�
=?0�׿@  @��\B��HC�33?0��<�@���B��@4z�                                    Bx��z  �          @�{?c�
�0��@���B���C�"�?c�
=�Q�@��HB�Q�@��
                                    Bx�   �          @��?:�H�&ff@z�HB�\C�9�?:�H=�G�@~{B��Az�                                    Bx��  "          @vff?   >�@q�B��=B*G�?   ?�
=@g�B��B��R                                    Bx�+l  �          @~�R?333=�G�@z=qB��{A�?333?Y��@tz�B���BK33                                    Bx�:  �          @s33?��׿�\@aG�B�L�C��f?���>8Q�@c33B�aHA(�                                    Bx�H�  
�          @s33?�Q��@b�\B�#�C�O\?�Q�>�@eB�@\                                    Bx�W^  �          @tz�?�\)��33@g�B�
=C�s3?�\)>�Q�@g�B��fA��
                                    Bx�f  T          @xQ�?��
����@n{B��qC��{?��
>�(�@l��B�p�A�p�                                    Bx�t�  "          @z�H?aG���(�@l(�B�=qC��?aG�>���@l��B�A��\                                    Bx��P  �          @�G�?��
>�{@w
=B���A��R?��
?�=q@n�RB�p�B9��                                    Bx���  �          @�(�?W
=�33@VffBaC�#�?W
=��{@k�B���C��
                                    Bx���  "          @�녾k�����?��
Ad��C�\)�k��qG�?�A�Q�C�=q                                    Bx��B  T          @�G��\)��  <�>��C�w
�\)����?h��A9��C�c�                                    Bx���  
�          @���=p���33������C��=�=p�����?L��A{C���                                    Bx�̎  �          @��\�8Q���Q�Ǯ��{C��\�8Q����?��@�Q�C���                                    Bx��4  	�          @��Ϳh������>�z�@X��C��)�h������?�p�Ah��C�p�                                    Bx���  �          @�  �fff��������p��C��{�fff��(�?&ff@�C�˅                                    Bx���  �          @��\�k���{�!G�����C�q�k���
=>��R@j�HC�|)                                    Bx�&  
�          @�=q��z���=q?�ffAIG�C�Ϳ�z�����?�Q�A��C~ٚ                                    Bx��  �          @������\��33?�G�A��C�o\���\�z�H@%Bz�Cu�                                    Bx�$r  T          @��
�xQ�����?�  A�G�C����xQ���z�@�A���C�l�                                    Bx�3  "          @�ff�n{��?��Au��C�aH�n{���\@{A���C��                                    Bx�A�  �          @���(����  ?�\)A~�\C��(����(�@�\AڸRC��q                                    Bx�Pd  �          @�\)�=p���  ?��Ap��C��ÿ=p�����@{AӮC�<)                                    Bx�_
  �          @�\)�Tz���\)?�ffAr�HC�f�Tz����
@�RAԸRC��H                                    Bx�m�  �          @�\)�fff����?s33A0Q�C��\�fff��Q�?�33A��C�Z�                                    Bx�|V  �          @�ff��ff��\)?�G�A<(�C��ῆff��?�Q�A��HC�P�                                    Bx���  �          @��R���
��  ?�  A9C�޸���
��ff?�Q�A�{C�xR                                    Bx���  T          @�\)�}p�����?��\A<��C�)�}p����R?��HA��C��
                                    Bx��H  T          @��c�
���?uA3�
C����c�
��ff?�33A�{C�O\                                    Bx���  T          @�33�\(���ff?W
=A Q�C�Ф�\(���p�?��
A��RC��H                                    Bx�Ŕ  T          @�=q������>Ǯ@��
C��{������  ?��A��HC�U�                                    Bx��:  �          @��
�L����ff?E�AffC��=�L����ff?�A�=qC��q                                    Bx���  �          @�
=�G���33?.{A�
C��\�G����?ǮA�33C���                                    Bx��  �          @���
=q��p�?z�@�
=C����
=q��ff?�p�A�{C�Z�                                    Bx� ,  �          @�{������?�@���C�)�������?��HA��C��=                                    Bx��  �          @�ff��{��z�?�@׮C���{��{?�A��C���                                    Bx�x  �          @���aG���33?!G�A�C����aG����
?\A�\)C�}q                                    Bx�,  �          @�33�@  ��  >�G�@�G�C�  �@  ��=q?��A���C�Ǯ                                    Bx�:�  "          @�녿Tz���p�?(��A��C�Y��Tz��|(�?�G�A��\C��                                    Bx�Ij  �          @�z�c�
�z�H?h��AL��C��{�c�
�i��?��HA�(�C�)                                    Bx�X  �          @�G���ff��p�?Y��A8z�C�箾�ff�y��?��HA�Q�C���                                    Bx�f�  �          @����k�����?�G�A[\)C����k��p  ?�A�  C��                                    Bx�u\  �          @�Q�8Q����
?E�A&�RC�
=�8Q��w�?�\)A���C���                                    Bx��  �          @�G��c�
��(�?8Q�A�\C���c�
�xQ�?���A�Q�C��\                                    Bx���  "          @�z�Q�����>B�\@ ��C����Q���p�?���Ag�
C�l�                                    Bx��N  
�          @�(��u��Q�>W
=@0  C����u���?�{Aj�HC�u�                                    Bx���  
�          @��R�������?�@�33C�ÿ�����\?�
=A��Cz�                                    Bx���  �          @�33���
��p�?+�A
=C�p����
��?���A�33C��                                    Bx��@  �          @�(������z�?k�A8��Cs3������H?�A��C~p�                                    Bx���  �          @�(���
=��33?��\AL  C~�쿗
=����?�
=A�33C}�                                    Bx��  "          @��H���R�l(�?��Ah��Cq� ���R�W�?�{A���Co�=                                    Bx��2  T          @�p���o\)?��
AXQ�Cp��[�?�A�  Cn�)                                    Bx��  �          @�\)�(Q��aG�?J=qA!�Ci8R�(Q��QG�?��
A�Cg(�                                    Bx�~  �          @�p��2�\�l��>���@�ffCh�q�2�\�aG�?�Q�Amp�Cg�f                                    Bx�%$  �          @��\�aG��Mp�>W
=@%�C^W
�aG��E�?fffA.�HC]=q                                    Bx�3�  
�          @��R�N{�Z=q������Cb���N{�h�ÿ&ff���Cdz�                                    Bx�Bp  "          @�z��4z��I����
=��Q�Cd&f�4z��_\)�����o33Cg{                                    Bx�Q  �          @��
��Q��Z=q��z����Cpk���Q��p  ��{�l��Cr�f                                    Bx�_�  "          @��ÿ�\)�P  �G��\)Cw���\)�j�H��G����RCy�                                     Bx�nb  "          @�ff�#�
�<���8Q��1z�C��ͼ#�
�`���
�H����C��\                                    Bx�}  �          @Y��?�p��u�:=q�m(�C�?�p��333�4z��bffC�H                                    Bx���  �          @b�\?�G�?(���@  �gA�
=?�G����
�Dz��p�HC��                                     Bx��T  "          @S�
?���>�G��3�
�vA�?����L���5�z�HC�&f                                    Bx���  "          @A�?}p�?\�  �LG�Bd
=?}p�?}p��!��tQ�B3�
                                    Bx���  �          @5�?O\)?�(����>�\B��\?O\)?��H�=q�kffBa�                                    Bx��F  T          @>{=L��@����1��B�(�=L��?����   �e=qB�=q                                    Bx���  �          @:=q>\)?��  �J��B���>\)?���&ff�}�B�.                                    Bx��  "          @1�=��
?�(��z��fG�B�=��
?k��%�B�aH                                    Bx��8  
�          @\(�>W
=?���J=q�B��>W
=?���XQ�B��                                    Bx� �  
�          @w�>.{?����i��33B�>.{>�33�u�¨W
B                                    Bx��  �          @qG�?c�
���Fff�^�C�z�?c�
�#�
�%�.�\C���                                    Bx�*  
Z          @e�?�  ����J=q�z�
C���?�  ��(��2�\�Nz�C�y�                                    Bx�,�  T          @`  ?��Ϳ�z��7��](�C��=?��������H�/�\C��{                                    Bx�;v  T          @\(�?O\)���1G��X{C�� ?O\)��H���'
=C�>�                                    Bx�J  "          @Vff?c�
�ٙ��.{�[\)C���?c�
�G�����+ffC��\                                    Bx�X�  �          @XQ�?�\����\)�B=qC���?�\�)����Q���C�u�                                    Bx�gh  �          @`��=�G��{�(��2{C�=q=�G��>{����z�C��                                    Bx�v  
�          @n�R=�\)�L�Ϳ��
��\C��R=�\)�a녿�  �}�C��=                                    Bx���  �          @p  >\)�g
=�xQ��p(�C��>\)�n�R��G���p�C��                                    Bx��Z  "          @l(�=�\)�i��>�@�
C���=�\)�aG�?xQ�Av{C��                                    Bx��   �          @b�\=�G��Vff?Y��Ac
=C��=�G��C�
?���A�Q�C��q                                    Bx���  
�          @_\)>��[�>8Q�@C33C�Ǯ>��R�\?xQ�A��\C��\                                    Bx��L  �          @_\)?(���Tz�>���@ָRC���?(���HQ�?�z�A���C��\                                    Bx���  "          @Z�H?k��O\)?\)A�\C��?k��AG�?�ffA���C�l�                                    Bx�ܘ  
�          @W
=?J=q�N{���	��C��=?J=q�J=q?!G�A/33C�
=                                    Bx��>  
x          @Mp�?k��C33�W
=�qG�C�g�?k��@��?�\A�RC��                                     Bx���            @\(�?�\)�E�?�  A��C�f?�\)�0��?�Q�A���C�{                                    Bx��  
(          @mp�?�z��Dz�?�  A���C�O\?�z��(Q�@�BG�C�\                                    Bx�0  �          @Q�?У��#33?��RA���C�S3?У��(�?�B=qC�e                                    Bx�%�  �          @e�?����4z�?�33A݅C�3?����ff@G�B��C�@                                     Bx�4|  �          @QG�@C�
��
=?��A�
=C�@C�
�#�
?���A��C�}q                                    Bx�C"  �          @N�R@B�\���H?s33A���C�t{@B�\��  ?��A�(�C���                                    Bx�Q�  �          @P��@HQ�8Q�?
=qA�C��f@HQ��?333AEC���                                    Bx�`n  T          @XQ�@J�H�z�H?��AG�C�g�@J�H�Q�?B�\AQ�C���                                    Bx�o  �          @X��@E���>��@�
=C��{@E���?B�\AP��C��                                    Bx�}�  �          @Z�H@:�H���H?uA�ffC��{@:�H��Q�?��
A�z�C��=                                    Bx��`  "          @W�@/\)��ff?��HA�33C�O\@/\)�h��?�p�A���C���                                    Bx��  �          @Y��@���  ?�
=A�p�C�33@���ff?��RB��C�XR                                    Bx���            @Tz�@G����H@z�BQ�C��{@G��.{@�
B0C��
                                    Bx��R  
�          @Q�?n{�{��p��G�C�ff?n{�\)>.{@��C�J=                                    Bx���  �          @c33�u�8Q���\�p�C�Z�u�R�\���
��z�C�o\                                    Bx�՞  �          @i��>�  �H�ÿ��
���C�@ >�  �^�R�u�u��C��                                    Bx��D  �          @g�>�  �Q녿�(�����C�.>�  �b�\��R��
C�                                    Bx���  �          @h�ý�Q��c�
�z���C�E��Q��fff>��R@�{C�Ff                                    Bx� �  �          @b�\��G��a논���C����G��[�?Tz�AZ�RC�R                                    Bx� 6  �          @]p����
�\(�    ��\)C��3���
�U?W
=Ab�\C���                                    Bx� �  �          @]p������[��k��w
=C�������XQ�?�RA%��C�|)                                    Bx� -�  �          @S�
�B�\�Q녾�����
=C�W
�B�\�P��>��HA��C�T{                                    Bx� <(  �          @5�?B�\��R�8Q��v�HC�}q?B�\�%����
��Q�C�.                                    Bx� J�  �          @H��?!G���?���B%�HC�@ ?!G�����@�HB\G�C���                                    Bx� Yt  T          @l��?:�H�ff@8��BQ�C���?:�H���@Tz�B�L�C�s3                                    Bx� h  T          @w
=?c�
�"�\@0��B6�
C��H?c�
��\@S�
Bl=qC�O\                                    Bx� v�  �          @|��?xQ��)��@0  B0��C���?xQ���@Tz�Bf
=C��3                                    Bx� �f  �          @|��?��R�0  @ ��B=qC�q?��R��@G�BRC���                                    Bx� �  T          @|��?\(��G�@\)B
��C���?\(��p�@<��BBQ�C���                                    Bx� ��  �          @~�R?&ff�Q�@B {C��3?&ff�(��@6ffB9�C��)                                    Bx� �X  T          @�  ?@  �C33@ ��Bz�C���?@  �z�@L��BR�
C��R                                    Bx� ��  �          @b�\?W
=��p�@>{Bd�C���?W
=�n{@S�
B��3C��                                    Bx� Τ  �          @z�H?Y���   @P��Ba�C��f?Y������@j=qB�aHC���                                    Bx� �J  �          @��?s33�33@L(�BP
=C�7
?s33��@j�HB�(�C���                                    Bx� ��  �          @�{?p���8Q�@8Q�B.33C�f?p���33@aG�BeffC�L�                                    Bx� ��  �          @�(�?��
�B�\@'
=B=qC�P�?��
��@S33BSz�C�"�                                    Bx�!	<  
�          @|(�?aG��>�R@�HB��C�G�?aG��  @G
=BP33C��
                                    Bx�!�  �          @}p�?fff�<(�@�RBz�C���?fff���@I��BS�C�!H                                    Bx�!&�  �          @z�H?s33�-p�@*=qB+{C��\?s33��Q�@QG�BbG�C�q                                    Bx�!5.  T          @�z�?z�H�!G�@HQ�BD�C��=?z�H��\)@k�Bz�
C���                                    Bx�!C�  T          @��?B�\�E�@!�BffC��?B�\�z�@O\)BT(�C�{                                    Bx�!Rz  �          @���?@  �K�@�B��C��R?@  �p�@G
=BI��C�p�                                    Bx�!a   T          @~�R?
=�Fff@�
B�
C�j=?
=���@B�\BK��C���                                    Bx�!o�  �          @���?5�7�@Y��BC�C���?5���@���B}��C�C�                                    Bx�!~l  �          @�Q�?B�\�:=q@k�BK
=C�T{?B�\��@��B�W
C�4{                                    Bx�!�  �          @�33?333�H��@g�BA=qC�J=?333�@�=qB{�HC�J=                                    Bx�!��  �          @�=q?+��N{@aG�B;�
C���?+���@�  Bv�C��                                     Bx�!�^  �          @���?E��Tz�@X��B3ffC��H?E���
@�z�BnG�C�/\                                    Bx�!�  "          @���?J=q�S33@X��B4
=C���?J=q��\@���Bn�HC���                                    Bx�!Ǫ  �          @�=q?B�\�6ff@^�RBF�C���?B�\��=q@��B��{C�\)                                    Bx�!�P  
�          @��?�p�����@{�B�8RC�!H?�p���Q�@�p�B���C��
                                    Bx�!��  T          @��
?�ff��p�@w
=By�C���?�ff�#�
@�B�Q�C�E                                    Bx�!�  
�          @��
?�=q� ��@mp�Bip�C�*=?�=q�s33@��B�W
C�b�                                    Bx�"B  T          @��?�����@p��Bp��C���?���O\)@��
B�=qC�XR                                    Bx�"�  �          @��?\��  @|(�B~
=C�G�?\���@�z�B�� C�=q                                    Bx�"�  �          @��H?�����@�G�Bv��C�
=?���:�H@���B��
C�&f                                    Bx�".4  �          @��H?}p����@vffBe33C��H?}p����@��B�L�C���                                    Bx�"<�  �          @���?��`  @5�B�
C�J=?��&ff@j�HBX�C���                                    Bx�"K�  �          @��>�
=�qG�@Q�B �C�.>�
=�?\)@S�
B>ffC�H                                    Bx�"Z&  "          @�\)?n{�:�H@P��B:�C���?n{��
=@{�Bu��C��3                                    Bx�"h�  "          @�33?@  �G�@S33B7Q�C���?@  �ff@���Bs�
C��q                                    Bx�"wr  �          @��R>\�E@L(�B6C��f>\�ff@z=qBuG�C�%                                    Bx�"�  T          @�Q�?k��+�@^�RBK�C�s3?k��У�@��\B���C��\                                    Bx�"��  T          @�Q�?�ff��H@i��BX�RC�� ?�ff����@�p�B��HC�4{                                    Bx�"�d  �          @�p�?��5@Tz�BD
=C�G�?�����@~{B�
=C�)                                    Bx�"�
  "          @�ff>Ǯ�Q�@AG�B)C�Y�>Ǯ�z�@s33Bi  C���                                    Bx�"��  �          @���>#�
�hQ�@/\)B
=C�8R>#�
�/\)@h��BS��C���                                    Bx�"�V  �          @��\>�  �_\)@<(�B \)C��>�  �"�\@r�\B`G�C��)                                    Bx�"��  �          @��
��=q�@i��Bd��C�� ��=q���R@��B�\C��                                    Bx�"�  �          @�=q>L���0  @S�
BH�
C�!H>L�Ϳ�(�@|(�B�� C�b�                                    Bx�"�H  �          @�(�<#�
�L(�@?\)B,z�C�{<#�
��R@p��Bm�C��                                    Bx�#	�  �          @��
>L���:�H@O\)B?��C���>L�Ϳ��@{�B�B�C��)                                    Bx�#�  �          @�33>�  �7�@P��BB  C�|)>�  ����@{�B�G�C�ٚ                                    Bx�#':  "          @��>�z��=p�@L(�B<=qC��\>�z��Q�@x��B|�
C�AH                                    Bx�#5�  �          @�33>�p��;�@L(�B<��C��{>�p���@xQ�B}�C�o\                                    Bx�#D�  �          @��H>���C�
@Dz�B4  C�k�>���z�@s�
Bu{C���                                    Bx�#S,  �          @�  >�G��E@O\)B8G�C��>�G��33@~�RBx�HC��                                    Bx�#a�  "          @���?�\�X��@=p�B#�C�C�?�\��H@s33Bc�
C���                                    Bx�#px  
�          @�Q�?
=�Y��@9��B �C��f?
=��@p  Ba{C��f                                    Bx�#  	�          @�\)?&ff�w�@%B�RC�� ?&ff�>�R@e�BF��C�#�                                    Bx�#��  �          @�Q�?#�
��=q@Q�A�(�C�}q?#�
�N{@\(�B9G�C���                                    Bx�#�j  �          @��?333����@#33B 33C��?333�G�@eBAQ�C�K�                                    Bx�#�  �          @�G�?&ff���@AυC�\)?&ff�^{@Mp�B)\)C�P�                                    Bx�#��  �          @��\?333��Q�@	��A�(�C���?333�]p�@R�\B+�C��R                                    Bx�#�\  "          @��\?G����@A�\)C�Y�?G��S�
@[�B5{C���                                    Bx�#�  �          @��\?333��(�@=qA�C���?333�P��@`  B9�\C��                                    Bx�#�  
�          @���?(����z�@A���C���?(���R�\@[�B6z�C��3                                    Bx�#�N  �          @�\)>���}p�@#�
B�C�j=>���B�\@fffBE��C�l�                                    Bx�$�  �          @��׾���u@4z�B�C�ٚ����7
=@s�
BTQ�C�t{                                    Bx�$�  �          @�����G��|��@  A�
=C�4{��G��HQ�@S33B:�C���                                    Bx�$ @  �          @��H>���j�H@Q�A�C�.>���8��@FffB;  C��                                    Bx�$.�  �          @��\>�\)�l(�@��A�G�C�'�>�\)�8��@K�B>Q�C��                                     Bx�$=�  "          @�=q?���tz�?�33A�G�C�l�?���Fff@<(�B+C�p�                                    Bx�$L2  
�          @���?�ff�mp�?��A�\)C�޸?�ff�AG�@333B$�HC���                                    Bx�$Z�  �          @�33?���h��@ffA�Q�C��?���7
=@E�B5Q�C��                                    Bx�$i~  @          @�z�?c�
�k�@�RA�(�C���?c�
�6ff@Mp�B<=qC��f                                    Bx�$x$  �          @�z�?Tz��g
=@B�C�}q?Tz��0��@S�
BC��C�e                                    Bx�$��  �          @�z�>��
�Vff@3�
B33C��
>��
��@j�HBc��C��{                                    Bx�$�p  
�          @�p�>�{�XQ�@4z�B��C��{>�{���@l��BcffC��q                                    Bx�$�  
�          @�\)?��fff@%B�C�|)?��*�H@b�\BQ��C��                                    Bx�$��  �          @��R?(���+�@^�RBNffC��
?(�ÿ\@��
B�.C�Ф                                    Bx�$�b  �          @��H?!G���@g�BcQ�C��f?!G�����@�z�B�{C���                                    Bx�$�  �          @�ff?J=q�Fff@,(�B \)C�/\?J=q�	��@`  Bc
=C�"�                                    Bx�$ޮ  "          @��R?u�AG�@G�B1�C���?u����@xQ�Br��C�!H                                    Bx�$�T  �          @�ff?u�:=q@Mp�B9Q�C�+�?u��@|(�Bz  C��                                    Bx�$��  �          @�ff?p���9��@Mp�B9��C�  ?p�׿�ff@{�Bz��C���                                    Bx�%
�  "          @�����\)�;�@333B.\)C�:ᾏ\)����@c�
Bt=qC��R                                    Bx�%F  �          @{�>Ǯ��
@S33Bfp�C�XR>Ǯ�u@qG�B�=qC�                                      Bx�%'�  �          @�Q�?��&ff@"�\B/Q�C���?��ٙ�@Mp�Bs��C�|)                                    Bx�%6�  "          @��H?&ff�n{?�  A���C��)?&ff�E@$z�B\)C��)                                    Bx�%E8  �          @�p�?   �g�?�p�A�C��?   �5@?\)B8  C��R                                    Bx�%S�  �          @�p�>��H�r�\?��A�(�C��>��H�G
=@.{B#�C�y�                                    Bx�%b�  �          @��H?\(��b�\?�
=AŅC��{?\(��7
=@+�B'��C�`                                     Bx�%q*  "          @��\>����S33@33B
��C�l�>�����@L��BQp�C��                                     Bx�%�  "          @z=q>��_\)?��A�
=C���>��6ff@"�\B%\)C�Ǯ                                    Bx�%�v  @          @s33>��c�
?���A���C�Ф>��C33@	��BffC�s3                                    Bx�%�  �          @w�>���Z=q?��
AۮC�*=>���,(�@/\)B5�\C��q                                    Bx�%��  
�          @o\)?�Q��>{?�z�A�Q�C��=?�Q���R@.�RB<C�                                    Bx�%�h  
(          @QG�@33��?�p�A���C��{@33��{?޸RB33C���                                    Bx�%�  @          @N{?���.�R?��\A�Q�C��?�����@33Bp�C�33                                    Bx�%״  �          @_\)?�p��Mp�>��@��HC�z�?�p��9��?���Ař�C�z�                                    Bx�%�Z  T          @p  ?У��U������33C��?У��R�\?(��A#�
C�0�                                    Bx�%�   �          @w
=?�=q�B�\?.{A,��C�}q?�=q�+�?˅AхC�%                                    Bx�&�  T          @y��@��,��?�A�(�C��@��   @#�
B%ffC�Y�                                    Bx�&L  �          @j=q@z���\?�=qA��C��@z����@p�B,��C�'�                                    Bx�& �  �          @_\)?�\)�
=q@	��B�
C�c�?�\)����@.{BO��C���                                    Bx�&/�  �          @e�?�
=���R@"�\B8(�C�ٚ?�
=���@B�\Bm�C�b�                                    Bx�&>>  �          @c�
?��\�
=@\)B5�C�~�?��\��(�@AG�Bop�C�R                                    Bx�&L�  �          @Y��?����*�H?�  A�  C���?��ÿ�p�@   B@�C�<)                                    Bx�&[�  �          @Y��?�G��%�?�A��C�/\?�G����R@
=qB#ffC��3                                    Bx�&j0  �          @L(�@ff��>8Q�@L(�C��@ff���H?W
=Av�RC��                                    Bx�&x�  r          @G�?��
���>��@�p�C���?��
�\)?��
A�Q�C�(�                                    Bx�&�|  6          @>�R?�Q���H�\)�)��C���?�Q���?#�
AI�C��q                                    Bx�&�"  
�          @/\)?���>B�\@�{C��)?���{?Tz�A�33C��
                                    Bx�&��  �          @HQ�?�Q�>W
=�$z��a��@�z�?�Q��R�   �Y
=C���                                    Bx�&�n  �          @&ff?�Q�G���{��C�?�Q쿗
=�����z�C�e                                    Bx�&�  �          @&ff?���p�׿�\)���C�!H?����z῝p���ffC��q                                    Bx�&к  �          @=p�@33�}p��\���
C�Y�@33��
=��\)��p�C��                                    Bx�&�`  "          @S33@#�
��
=>�
=@�p�C���@#�
��Q�?��\A���C�T{                                    Bx�&�  
�          @U�@(���>�Q�@���C��{@(�����?�G�A�z�C�w
                                    Bx�&��            @R�\@�R��þ.{�8��C���@�R��?
=qA\)C��                                    Bx�'R  h          @]p�@.�R��������C�L�@.�R��
>�G�@��C��                                     Bx�'�  �          @Z=q@'
=�
�H��Q��=qC��@'
=�?�RA(Q�C��\                                    Bx�'(�  T          @_\)@<�Ϳ�ff��33����C�Ff@<�Ϳ���>��@���C�*=                                    Bx�'7D  �          @AG�@2�\�8Q�5�[�
C���@2�\�h�þ�(��p�C���                                    Bx�'E�  r          @l(�@'���R�����C�N@'��"�\>�  @|(�C���                                    Bx�'T�  5          @n�R@�\�9��>�  @x��C�/\@�\�)��?�Q�A�  C�l�                                    Bx�'c6  �          @p��?�p��>�R?c�
A^�\C��f?�p��!G�?���A��C�
=                                    Bx�'q�  "          @u@p��0  >��H@���C��@p��(�?�\)A���C��f                                    Bx�'��  T          @p  @\)�{��������C���@\)�,�;�  �w�C�^�                                    Bx�'�(  
�          @j�H@#�
�
�H��Q���\)C��
@#�
�!G��!G�� Q�C���                                    Bx�'��  "          @���@7��@�׿.{�(�C��)@7��C�
>�Q�@�z�C���                                    Bx�'�t  �          @�\)@4z��]p����Ϳ��C���@4z��R�\?��AZ�RC�G�                                    Bx�'�  �          @�ff@(Q��c�
�����tz�C�9�@(Q��\��?h��A=�C���                                    Bx�'��  �          @�
=@!G��c�
�J=q�%�C��=@!G��g�>�G�@��RC�p�                                    Bx�'�f  �          @��
@���]p�?uAS�C���@���<(�@�
A�{C��                                    Bx�'�  �          @�Q�@.�R�/\)?ǮA��C�s3@.�R��
@��B�RC�~�                                    Bx�'��  �          @�
=@�\�P  ?˅A�  C�\@�\�!�@%BG�C�u�                                    Bx�(X  �          @�
=?�=q�^{?��A֏\C�u�?�=q�(��@<(�B3�\C�\)                                    Bx�(�  "          @�p�?&ff�:�H@8��B/�HC�L�?&ff���
@l��B{��C��                                    Bx�(!�  �          @�\)>Ǯ�:=q@AG�B733C�޸>Ǯ���H@tz�B��\C�|)                                    Bx�(0J  
�          @���>�Q��j�H@A�(�C��{>�Q��/\)@L��BD��C��f                                    Bx�(>�  �          @��>aG��g�@��A���C��>aG��*�H@O\)BI�\C�c�                                    Bx�(M�  �          @�Q�>���Tz�@%B��C��>���\)@dz�Be�C�.                                    Bx�(\<  �          @�  ?�\)�=p�@4z�B&��C�T{?�\)����@j=qBn�RC��q                                    Bx�(j�  �          @�ff?���{@R�\BSffC���?�녿u@vffB���C�f                                    Bx�(y�  
�          @���@
�H���@X��BR��C���@
�H���@j=qBmG�C���                                    Bx�(�.  �          @�?�{�   @W�BY=qC�)?�{�8Q�@vffB�\)C��)                                    Bx�(��  T          @��?��
�Ǯ@hQ�Bz�
C��f?��
�aG�@|(�B�(�C��                                    Bx�(�z  
�          @�>��H�+�@HQ�BC�HC�'�>��H��Q�@w�B�
=C�\)                                    Bx�(�   T          @������p  ?�=qA��RC~
=�����C�
@"�\B\)Cz�3                                    Bx�(��  �          @��=��
�j=q@z�Bz�C���=��
�(Q�@\(�BRQ�C��                                    Bx�(�l  T          @��>���b�\@$z�B
=C�C�>����@h��B_G�C���                                    Bx�(�  �          @���?@  �E�@B�\B/G�C�޸?@  ��=q@z�HB|C�*=                                    Bx�(�  �          @���?���>{@7�B)33C��f?�����@n{Bs\)C�C�                                    Bx�(�^  �          @�G�?�  �C�
@4z�B%
=C�
=?�  ���@mp�Bpp�C���                                    Bx�)  "          @�=q?�R�G�@:=qB)�C���?�R��z�@tz�By{C��                                    Bx�)�  "          @�G�?   �:�H@E�B8Q�C���?   ��z�@y��B��C�W
                                    Bx�))P  �          @�=q>�33�(��@XQ�BO�C��>�33���@��HB���C���                                    Bx�)7�  
�          @��H��G��C33@\��BBG�C��)��G����@���B�(�C�q                                    Bx�)F�  "          @�Q�<��5@\��BJ�C�Z�<�����@�\)B�33C���                                    Bx�)UB  �          @�>����c�
@?\)BffC�j=>�����\@���BqffC��H                                    Bx�)c�  �          @�>��R�Dz�@\��B@��C��>��R��33@��B�Q�C�W
                                    Bx�)r�  �          @��#33��G�@C�
B<=qCN.�#33��Q�@S33BQ�C6�                                    Bx�)�4  �          @��Fff���H@/\)B$ffC<���Fff?�\@.�RB$  C*�{                                    Bx�)��  
(          @�{�[�>�@��BQ�C1���[�?u@(�A�z�C$L�                                    Bx�)��  T          @��_\)?\)@
�HA�ffC*�
�_\)?�ff?�=qA�G�C�)                                    Bx�)�&  
�          @�(��`��?���?�A�Q�C#��`��?��?��A��C�                                    Bx�)��  T          @��R�k�?��?��A�Q�C+k��k�?�Q�?ǮA��C"�                                    Bx�)�r  
�          @��R�j�H��Q�@�A��HC9�\�j�H>Ǯ@�A�{C-�f                                    Bx�)�  T          @�{�j=q���@ ��A�Q�C6\)�j=q?z�?�Q�A�z�C+                                      Bx�)�  �          @��\�xQ콸Q�?�33A���C5B��xQ�?��?�AŮC++�                                    Bx�)�d  "          @�G��~�R��
=@��A���C:��~�R>\@	��A��
C.�
                                    Bx�*
  �          @�33�x�ÿ!G�?�=qA�33C=&f�x��=�\)?�
=A�G�C2�                                    Bx�*�  
�          @�����Q�8Q�?�
=A��HC>)��Q�<�@�
A�p�C3�
                                    Bx�*"V  
(          @�G��}p���ff@�A�
=C:u��}p�>�p�@(�A���C.�3                                    Bx�*0�  �          @�  �p�׾��@�B�HC:B��p��>��H@=qB��C,��                                    Bx�*?�  
�          @�\)�mp��W
=@�RB(�C7@ �mp�?333@��B�C)=q                                    Bx�*NH  �          @�
=�p  ��@��Bp�C4ff�p  ?Y��@\)A�Q�C'G�                                    Bx�*\�  �          @��dz�=�@%B�RC2)�dz�?�ff@�B�C#��                                    Bx�*k�  �          @�p��c�
��p�@#33B�HC:��c�
?z�@ ��Bp�C*�                                     Bx�*z:  �          @����dz�.{@\)B =qC6��dz�?(��@	��A���C)�                                     Bx�*��  �          @�=q�G���z�@'�B��CLB��G���33@=p�B,�HC:p�                                    Bx�*��  "          @����L(���  @%B��CIk��L(��W
=@7�B'�
C7�
                                    Bx�*�,  T          @���U����@{B{CF���U�\)@-p�B�C6ff                                    Bx�*��  T          @�=q�C�
�}p�@7�B&�CE�q�C�
>�@A�B2�C1��                                    Bx�*�x  �          @��R�G��J=q@0  B"  CBG��G�>�=q@6ffB)(�C/                                      Bx�*�  T          @����U����@�B
=CI�q�U��
=@'
=B33C;�                                    Bx�*��  �          @�G��N{��@#�
BQ�CG�3�N{�\)@3�
B$�\C6xR                                    Bx�*�j  �          @�33�C33�33@\)A��\CU�\�C33����@5�B$��CG\)                                    Bx�*�  
�          @�33�,(��'�@p�A�C`8R�,(���{@AG�B/�CR��                                    Bx�+�  "          @��H�,��� ��@�\A��\C^��,�Ϳ�p�@C�
B3�CP�                                    Bx�+\  �          @��\�;��@
�HA��CZ�{�;���{@8��B'33CL�                                    Bx�+*  �          @����E�\)?�A�33CW���E��\)@'
=B�HCK�H                                    Bx�+8�  "          @����<���33@33A�(�CZ  �<�Ϳ�\)@1G�B!�CL�3                                    Bx�+GN  "          @����>{�G�@A�\)CYY��>{����@1�B"(�CL                                    Bx�+U�  
�          @����0���G�@�
B��C[c��0�׿��R@?\)B2�CL#�                                    Bx�+d�  �          @���<(��Q�@��A�CW�R�<(���33@5B'�CIk�                                    Bx�+s@  �          @����(����\@��B  C\ٚ�(�ÿ���@HQ�B<p�CLh�                                    Bx�+��  	�          @���%��33@"�\B=qCZz��%��s33@G�BBz�CH(�                                    Bx�+��  �          @�=q��(��ٙ�@U�BP=qC\�{��(����
@n�RBw�\C=0�                                    Bx�+�2  	�          @��
��p��G�@Q�BD�HCaz��p��!G�@r�\Bu�CE��                                    Bx�+��  �          @�(��!G���@E�B3=qCW���!G���\@b�\BW�C?s3                                    Bx�+�~  T          @��H��Ϳ��@FffB6�CX=q��;��H@c33B[��C?@                                     Bx�+�$  
�          @�
=�   ���
@N�RB9��CWu��   ��
=@j�HB]\)C=�{                                    Bx�+��  �          @�{��R��@P  B=�RCU���R���R@h��B^33C;
                                    Bx�+�p  T          @�(������R@U�BE�HCS� ����G�@i��Ba=qC6�
                                    Bx�+�  "          @�
=� �׿��H@_\)BM��CM��� ��>W
=@l(�B^��C/&f                                    Bx�,�  �          @���
=��p�@XQ�BL\)C[8R�
=���R@q�Br33C<8R                                    Bx�,b  
Z          @�G��p����@J=qB@
=C[  �p���ff@fffBg�
C?k�                                    Bx�,#  
�          @����   ��\)@B�\B6z�CT��   ����@[�BVC;�=                                    Bx�,1�  T          @���  ��R@8��B"�
Cc�=�  ��Q�@g�B[ffCO��                                    Bx�,@T  �          @�ff��녿�(�@p  Bm�Cf�f��녾\)@��
B�p�C9��                                    Bx�,N�  T          @�=q���׿У�@��HB�B�CkJ=����=�\)@��B�.C0��                                    Bx�,]�  �          @������R���H@�33B�  CeǮ���R>k�@�33B�C)^�                                    Bx�,lF  "          @��\�#�
��(�@�z�B�
=C����#�
?�@���B��{B��
                                    Bx�,z�  �          @���>����@�33B�� C��3>��#�
@�Q�B��HC��=                                    Bx�,��  "          @�{>B�\�0��@��
B�W
C��3>B�\?�33@���B�.B�u�                                    Bx�,�8  �          @�=q>����\@���B�aHC�H>��?���@��B�aHB���                                    Bx�,��  �          @�z�=L�Ϳ+�@�=qB�ffC��
=L��?�z�@�
=B��HB�z�                                    Bx�,��  �          @�p�>L�Ϳ
=@���B�p�C�XR>L��?�(�@���B���B��=                                    Bx�,�*  �          @���>�  ?�=q@���B��\B�Q�>�  @\(�@\��B3��B���                                    Bx�,��  T          @�ff>W
=�W
=@��B�z�C�Q�>W
=?�=q@���B�#�B���                                    Bx�,�v  �          @�ff?�\>k�@���B�G�A���?�\?��
@g
=B{�\B�.                                    Bx�,�  
�          @�G����
�H��@FffB,�RC{�
���
��
=@��\B�� Cnu�                                    Bx�,��  �          @��ÿ��H�u�@A�  C|� ���H�,��@Z�HBD�Cu�3                                    Bx�-h  T          @�
=�����~�R?��RA��\C{�׿����E�@;�B$�
Cv                                    Bx�-  "          @u?:�H��?   A�RC�\?:�H���
?0��B"�RC�f                                    Bx�-*�  "          @���@I��@�H������\)B�
@I��@�
������{B�H                                    Bx�-9Z  T          @�G�@<��@'
=�W
=�?�B&Q�@<��@z��ff��{B�\                                    Bx�-H   "          @�G�@9��@+��@  �+�B+(�@9��@
�H��  ���BQ�                                    Bx�-V�  T          @}p�@�H@G
=��{���
BP�@�H@.�R�������BA��                                    Bx�-eL  "          @~�R@�
@Y��=L��?B�\Bk(�@�
@I����ff��=qBc
=                                    Bx�-s�  �          @g�@z�@7
=���G�BX��@z�@%���  ��p�BM{                                    Bx�-��  �          @E@z�?�Q���5�B(�@z�?��ÿ�����33A�\)                                    Bx�-�>  �          @8Q�@�R?�=q�n{����A��R@�R?!G����
�ԏ\Ab�R                                    Bx�-��  "          @`��@=p�?�ff�z�H���HA���@=p�?��
�\��z�A���                                    Bx�-��  �          @_\)@9��?�p��W
=�`Q�A�  @9��?�  ���H�Ə\A�(�                                    Bx�-�0  
�          @a�@Dz�?˅�(���-��A���@Dz�?�Q쿞�R��33A��                                    Bx�-��  "          @\(�@8��?�(��\)��
A���@8��?��Ϳ�����p�A��                                    Bx�-�|  �          @<(�@�?�녿�Q���\)Aԣ�@�>�׿�\�=qA:ff                                    Bx�-�"  
�          @ff?���\(��^�R��  C�]q?����녾�ff�TQ�C��                                     Bx�-��  �          @\)?�p���33�Q���=qC��?�p����׾�z���p�C��=                                    Bx�.n  �          @'
=?��?��׿�=q���BQ�?��?Tz�����HAƸR                                    Bx�.  "          @c33<��^{�.{�0Q�C�=q<��\��?G�AK33C�=q                                    Bx�.#�  �          @*=q?&ff�33��(����C���?&ff����z��أ�C���                                    Bx�.2`  �          @0  ?�����xQ���
=C�޸?��녾#�
�W�C�
                                    Bx�.A  �          @,��?ٙ����^�R����C�` ?ٙ���\�L�Ϳ�{C��3                                    Bx�.O�  �          @%�?��ÿ�����R�yp�C�h�?�����>aG�@�\)C��f                                    Bx�.^R  �          ?�(�?aG��=p�?W
=B�C�� ?aG���{?���BB  C�b�                                    Bx�.l�  �          ?�?��;\?���BGp�C���?���>W
=?�z�BN\)A0��                                    Bx�.{�  
�          ?�{?�=q�L��?�(�BVp�C��?�=q?��?�{BA��A֏\                                    Bx�.�D  
�          @{?}p���G�?��HB|\)C��=?}p�?333?�=qBb=qB{                                    Bx�.��  
�          @Dz�?\(���p�@8��B�.C�u�?\(�?W
=@2�\B�Q�B1��                                    Bx�.��  �          @8Q�?��
�#�
@%�B�� C��q?��
>�@'�B�G�A�G�                                    Bx�.�6  
�          @r�\@�
���
@0  B=�C�  @�
>8Q�@;�BN��@���                                    Bx�.��  
�          @\��?�녿�p�@#�
BMQ�C���?�녽�\)@5BoC���                                    Bx�.ӂ  "          @\��?��ÿ��@,(�BMp�C��=?���=u@:�HBhG�?��
                                    Bx�.�(  "          @tz�?�Q쿀  @HQ�B\z�C�h�?�Q�>�p�@P��Bk(�A.ff                                    Bx�.��  �          @mp�?����\@>�RBUQ�C���?���Ǯ@\(�B���C���                                    Bx�.�t  
�          @P��?�����@0  Bf��C�b�?���\)@E�B�W
C�/\                                    Bx�/  T          @\(�?�Q쿥�@8Q�Bj��C�N?�Q�<#�
@J=qB�>�
=                                    Bx�/�  
�          @vff@H��?}p�?��RA�
=A��@H��?�(�?�33A�\)A噚                                    Bx�/+f  T          @r�\@E�>��@��B	�A	@E�?��?�  A�33A�                                    Bx�/:  �          @Dz�@G���ff?��B��C�Y�@G�>�p�?�z�B��A\)                                    Bx�/H�  �          @1G�?�z᾽p�?�B2Q�C��{?�z�>�?�33B/�AXQ�                                    Bx�/WX  T          @8��?�(���33@��BJ(�C�G�?�(�?�@BD{A��                                    Bx�/e�  �          @-p�?�׿�?�\)B/(�C�N?��>���?�z�B4�A
=                                    Bx�/t�  �          @
=?��׿(�?��RB2�C�%?���=�\)?���BEG�@1�                                    Bx�/�J  
�          ?��?h�þ�ff?L��B��C��H?h�ý��
?k�B5
=C���                                    Bx�/��  �          @=q?��R���\?��B"
=C���?��R��\)?���BI{C���                                    Bx�/��  
Z          ?�ff?n{��
=?��\B�C�!H?n{�&ff?�BM�RC���                                    Bx�/�<  �          @G�?�
=���\@(��Bm��C���?�
=>.{@5�B�Q�A�                                    Bx�/��  "          @7�?��R�k�@��BQG�C�"�?��R=�G�@(�BjQ�@�\)                                    Bx�/̈  �          @G�?��Ϳ�(�>.{@��C���?��Ϳ��\?E�A���C���                                    Bx�/�.  "          @�R?��
��G�?�=qB>G�C��\?��
���?���Bp�
C��                                    Bx�/��  "          @,(�?s33���@��B}�C��?s33>���@33B��=A���                                    Bx�/�z  �          @j=q=u>.{@h��B�ffB�=u?�
=@N�RBz=qB���                                    Bx�0   �          @j�H��>�Q�@g
=B��CJ=��?�@G�Bi��BӅ                                    Bx�0�  �          @g��(�?p��@X��B�ffB����(�@�@+�BBz�B��
                                    Bx�0$l  �          @   ?}p��G������(�C�)?}p��G���Q���C���                                    Bx�03  "          ?��?(��k�����=qC���?(��\��=q���C��f                                    Bx�0A�  T          @AG�>��ÿ�\)?�{BAffC��{>��ÿ.{@
=B�aHC�f                                    Bx�0P^  
�          @%�?���
=?+�A��C�aH?�Ϳ�\)?�  B$C�n                                    Bx�0_  
�          @0�׾8Q���\?��
B$
=C�w
�8Q쿌��@�RB��RC�N                                    Bx�0m�  "          @l��?�  �4z�?���B{C��?�  ��p�@=p�BX�HC�ٚ                                    Bx�0|P  
�          @u?���:=q?�Q�A�=qC��?����@?\)BN�C�,�                                    Bx�0��  �          @u?޸R�:=q?޸RA���C�w
?޸R��33@3�
B=�
C�K�                                    Bx�0��  T          @���@
=�>�R?��HA��HC�33@
=�(�@
=B  C���                                    Bx�0�B  "          @��
@'
=�'
=?��A��HC��H@'
=����@.�RB&�
C�>�                                    Bx�0��  T          @��@5�
=q@�\A�  C�c�@5��=q@0��B){C��\                                    Bx�0Ŏ  �          @\)?��+�@
=qB
=C�?���  @FffBS(�C�'�                                    Bx�0�4  �          @���@#�
��R@
=A�33C�p�@#�
��\)@7
=B6��C�4{                                    Bx�0��  
Z          @|��@Mp���z�?�p�A��\C��@Mp��W
=@�B�\C�&f                                    Bx�0�  �          @u@>{��?��HA�ffC�7
@>{��G�@Q�B�C��q                                    Bx�1 &  �          @g�@0�׿5@(�Bz�C���@0��>�z�@�\B(�@�33                                    Bx�1�  �          @dz�@!녿��\@BC���@!녾�  @(�B/  C�&f                                    Bx�1r  �          @o\)@(�ÿ��H@p�B��C�|)@(�þ�p�@'�B2(�C���                                    Bx�1,  �          @p��@z����?�=qA��\C��f@zΐ33@$z�B333C��
                                    Bx�1:�  �          @w
=?У׿�@QG�Bj  C�)?У�>�33@]p�B���AA                                    Bx�1Id  �          @w
=@�ÿ�=q@8��BC��C��@��=L��@K�B`{?�(�                                    Bx�1X
  
�          @w�@(���
=@)��B,��C�˅@(����@@  BK�C�>�                                    Bx�1f�  �          @tz�@2�\��ff@
=B��C�}q@2�\���H@$z�B(�HC��                                    Bx�1uV  �          @y��@A녿�\?��HA�p�C��H@A녿\(�@33B��C��                                    Bx�1��  "          @\)@@  ��(�?�p�A��C�@@  �333@!�B�C�n                                    Bx�1��  "          @|(�@;���p�?�p�A�C��R@;��333@!�B G�C�C�                                    Bx�1�H  T          @~{@;�����?�A�C��R@;��W
=@!�B�HC�H                                    Bx�1��  �          @�  @K���\)?��RA��C�S3@K���p�@��B�C��f                                    Bx�1��  �          @�Q�@S33��Q�?���A���C��@S33�aG�@G�B	��C��                                    Bx�1�:  �          @���@K��#�
@�B�C��3@K�?h��@  B��A�                                    Bx�1��  �          @���@J�H����@   BffC�J=@J�H?O\)@Q�B
=Af�R                                    Bx�1�  
�          @z�H@:�H���
@#�
B$ffC���@:�H?Q�@��B��A|(�                                    Bx�1�,  "          @~{@C�
�u@!G�B��C�p�@C�
?�ff@33BffA���                                    Bx�2�  �          @��\@C�
��G�@+�B$�C��@C�
?���@p�BA��                                    Bx�2x  
m          @���@J=q����@\)B�
C��@J=q?}p�@�\B
Q�A��                                    Bx�2%  
Z          @��@W�=#�
@��B��?B�\@W�?�  ?��HA�33A�z�                                    Bx�23�  �          @���@n{>��H?�G�A���@��@n{?���?��A���A�                                    Bx�2Bj  
�          @��@7�    @�
B(�<#�
@7�?�G�@z�B	�A�33                                    Bx�2Q  T          @���@G��:�H@c33BkG�C��@G�?fff@`��BgG�A�                                    Bx�2_�  �          @���@U?�=q?�
=Aȣ�A��@U?�
=?n{AX(�A�R                                    Bx�2n\  "          @�Q�@Z=q?��\?�A�33A�
=@Z=q?�\)?p��AZ=qA�{                                    Bx�2}  �          @���@Tz�?��?���A���A�{@Tz�@�
?�ffAo\)A��R                                    Bx�2��  �          @�Q�@S33?�33?���A�G�A��@S33?��?�  A�
=A�z�                                    Bx�2�N  �          @��
@J=q?�G�?�Q�A�A�\)@J=q@(�?xQ�AYB��                                    Bx�2��  
�          @�p�@R�\?�Q�?��
A�=qA�Q�@R�\@�H?�\@�{B�                                    Bx�2��  T          @�p�@[�?�33?��A�=qA�Q�@[�@�\>�{@��RB�H                                    Bx�2�@  �          @��H@2�\?�\)@�B��A�\)@2�\@   ?�z�A�Q�B'\)                                    Bx�2��  "          @��?s33?B�\@}p�B��RB�\?s33@�@N{BK�\B�33                                    Bx�2�  �          @��
?��?p��@q�B�W
BG�?��@!G�@>�RB:{B{�\                                    Bx�2�2  �          @���>k�?G�@{�B�� B�G�>k�@(�@J�HBQ=qB�=q                                    Bx�3 �  T          @]p�?Tz�?k�@L(�B��=B@�?Tz�@\)@��B6�HB��\                                    Bx�3~  �          @|(�?�z�?�=q@_\)B|��BC�?�z�@/\)@"�\B"
=B�                                      Bx�3$  T          @dz�?˅?��@2�\BN(�B1  ?˅@'�?�A���BkQ�                                    Bx�3,�  T          @r�\?�  ?�Q�@9��BOB��?�  @%�?��HB p�B_�                                    Bx�3;p  T          @�z�@\)@'
=?�A��B9p�@\)@J=q>�@�(�BN�                                    Bx�3J  
�          @�
=@(Q�@%�@G�A�(�B1�
@(Q�@Mp�?(��AG�BJ                                    Bx�3X�  �          @��@!�@;�?��HA��HBD�@!�@J=q�aG��G
=BM�                                    Bx�3gb  �          @�@p�@&ff@��B{BF�@p�@X��?z�HAY�Bc��                                    Bx�3v  
�          @��R@�
@!�@�BB>  @�
@S33?uAU��B[                                    Bx�3��  �          @�33?��@J=q?��HA�  Bl?��@e=�?�  ByQ�                                    Bx�3�T  �          @�{?޸R@e�?�ffA��HB�#�?޸R@q녾�(���
=B���                                    Bx�3��  
�          @�33?�  @l��?��RA��\B�\)?�  @~{���
��B��                                     Bx�3��  �          @�Q�@ ��@w
=?�Ap(�By@ ��@~{�.{��B|z�                                    Bx�3�F  T          @��H?�\)@���?�@�B�ff?�\)@|�Ϳ�����ffB�8R                                    Bx�3��  �          @�z�?ٙ�@���>�z�@q�B�8R?ٙ�@n�R��ff��p�B���                                    Bx�3ܒ  T          @�\)@	��@^�R?^�RA?�BiQ�@	��@`  �J=q�-p�Bi�H                                    Bx�3�8  T          @��@=q@K�?�33A�(�BSQ�@=q@^{�8Q����B\                                    Bx�3��  	�          @���@,��@:=q?���A�ffB<��@,��@S�
=�Q�?�p�BK33                                    Bx�4�  "          @��R@:=q@Q�?�\)A��
B{@:=q@>{?��A  B6=q                                    Bx�4*  "          @�p�@G
=@�?���A��
Bz�@G
=@+�?0��A��B#=q                                    Bx�4%�  T          @��@>�R?�(�@Q�A�=qB33@>�R@.{?�G�Aa�B)�                                    Bx�44v  �          @��@7
=?˅@!G�B�A��@7
=@#33?��A��RB&�                                    Bx�4C  �          @���@;�?��
@N�RB5(�A��R@;�@&ff@�
A��
B&G�                                    Bx�4Q�  �          @�(�@I��?��H@:�HB�
A�p�@I��@5?�A�p�B(33                                    Bx�4`h  
�          @�
=@QG�@G�@.{B{A��
@QG�@A�?�G�A��B+\)                                    Bx�4o  T          @��\@@��@33@1�B��B��@@��@E�?ǮA���B6�R                                    Bx�4}�  �          @��@2�\@(�@)��B\)B$�H@2�\@XQ�?�G�A�z�BI�H                                    Bx�4�Z  �          @�(�@!�@%@��BffB6Q�@!�@Z�H?��\AV�RBU�R                                    Bx�4�   �          @���@'
=@8Q�@�A�Q�B?z�@'
=@_\)?   @��BT�H                                    Bx�4��  T          @�{@5�@C�
?�p�A�{B=�@5�@Y����\)�p��BH��                                    Bx�4�L  �          @�  @�H@]p�>k�@E�B\=q@�H@L(���\)��
=BS=q                                    Bx�4��  �          @�Q�@=q@U?�{Aq�BX��@=q@^�R�����
B]33                                    Bx�4՘  �          @��H@/\)@R�\=�\)?}p�BI  @/\)@>{��
=��  B=(�                                    Bx�4�>  T          @�@E�@_\)>Ǯ@��\BB33@E�@R�\���R�up�B;ff                                    Bx�4��  �          @�z�@AG�@^�R>��@�(�BD{@AG�@S�
��z��h  B>�\                                    Bx�5�  "          @��@333@e?(�@�p�BPQ�@333@^�R����[�BL�                                    Bx�50  �          @��R@#�
@_\)?�  AO33BW
=@#�
@c�
�333�{BY33                                    Bx�5�  T          @�@   @fff>�@��B\@   @[����H�~�RBWff                                    Bx�5-|  �          @�p�@0��@[���\)�uBL��@0��@AG��У����B>\)                                    Bx�5<"  
�          @���@$z�@O\)>��R@�\)BNz�@$z�@A녿��H��p�BF�                                    Bx�5J�  �          @���@�\@Y��?!G�A�B_�@�\@S�
�}p��]��B]�                                    Bx�5Yn  T          @��@��@Vff?&ffA�RBY�H@��@Q녿u�TQ�BWff                                    Bx�5h  �          @�\)@333@HQ�>���@�G�B@�@333@;������y�B9\)                                    Bx�5v�  �          @���@;�@8Q�?�G�A]�B2�@;�@AG���
=��ffB7�                                    Bx�5�`  �          @���@(Q�@I��?���A�  BHz�@(Q�@i��>L��@#�
BX�R                                    Bx�5�  �          @�\)@8��@N�R?���A^�RB@��@8��@W
=����  BE=q                                    Bx�5��  �          @�  @+�@9��?�p�A�
=B<@+�@^�R>�(�@��BQ�\                                    Bx�5�R  T          @�@#33@5�@  A�ffB?�
@#33@b�\?333A��BY
=                                    Bx�5��  "          @��@(Q�@B�\@G�A�BD�
@(Q�@hQ�>���@�33BX�\                                    Bx�5Ξ  
�          @�{@#�
@6ff@�A��B@33@#�
@b�\?!G�A�HBX=q                                    Bx�5�D  T          @�ff@(Q�@#33@   B	(�B0G�@(Q�@Z=q?�ffAZ�RBQff                                    Bx�5��  "          @u@{?��
@*�HB,�RB@{@3�
?�ffA���BN��                                    Bx�5��  �          @j=q?�\?�(�@:�HBV��B
(�?�\@�@�B\)BW�                                    Bx�6	6  �          @a�?��?�@&ffB?(�B��?��@��?�33A�33BQ�
                                    Bx�6�  "          @��H@#�
@{@z�A��HB/�H@#�
@I��?0��A\)BKQ�                                    Bx�6&�  �          @���@�R@Dz�?�Q�A�\)BLz�@�R@`��=��
?�{B[                                      Bx�65(  �          @�z�@�@b�\?=p�A�B`�H@�@^�R�z�H�QG�B_{                                    Bx�6C�  "          @dz�?�G�@G�=�G�?�Q�Br{?�G�@3�
�����ffBg��                                    Bx�6Rt  �          @aG�?�Q�@E�<#�
>W
=Bu{?�Q�@/\)��z���p�Biff                                    Bx�6a  
�          @g
=?�33@K��L���L��Bz{?�33@.�R��33��ffBk{                                    Bx�6o�  �          @X��?�G�@Dz�&ff�0(�B�k�?�G�@�H���R�ffBzG�                                    Bx�6~f  
�          @^{?�@7��B�\�K�Be(�?�@p����R�ϮBTz�                                    Bx�6�  �          @tz�@G�@L(�>W
=@L��Bf�R@G�@:�H�����p�B]��                                    Bx�6��  
�          @xQ�@G�@C33>��
@��BU\)@G�@6ff������33BM��                                    Bx�6�X  �          @~�R@'�@;�=#�
?(�B@�R@'�@'���������B3                                    Bx�6��  �          @�  @�@_\)>��@�(�B_ff@�@S�
������\)BY                                    Bx�6Ǥ            @���@p�@\��>��@�\)BeQ�@p�@O\)��p����B^�                                    Bx�6�J  g          @��\?�=q@c33?G�A0��Bz�?�=q@`  �s33�Y�By�\                                    Bx�6��  �          @�\)@Q�@^�R?���Ak�
BjG�@Q�@e��+���Bm(�                                    Bx�6�  T          @��@��@C33?�z�A�G�BO�R@��@O\)�Ǯ����BV\)                                    Bx�7<  T          @��@,(�@C�
>��R@�=qBB�@,(�@6ff��33����B:�                                    Bx�7�  T          @hQ�@*�H@�þ\����B'��@*�H?�Q쿹����\)BQ�                                    Bx�7�  T          @fff@;�?�p����\��{A�z�@;�?�����H��Q�A�ff                                    Bx�7..  �          @q�@�@0  ���
��Q�BA�H@�@�H��ff��B3p�                                    Bx�7<�  
�          @z=q@%?����33�33A�33@%>��#�
�1
=A�
                                    Bx�7Kz  "          @\)@!�@>{��\)����BF\)@!�@&ff������(�B6��                                    Bx�7Z   "          @�{@�\@U?^�RAB{B^ff@�\@W
=�J=q�/
=B_{                                    Bx�7h�  T          @�?��@QG�?�G�A�Bp(�?��@mp�<�>�(�B|Q�                                    Bx�7wl  �          @���?�  @Z�H?�Q�A�=qB{�R?�  @l�;��
��  B�ff                                    Bx�7�            @�{?�
=@XQ�?��RA��Bq  ?�
=@c�
���أ�Bv=q                                    Bx�7��  "          @���?���@`  >�=q@z=qBs�\?���@N�R�����{Bk\)                                    Bx�7�^  T          @���?���@^�R?L��A733ByG�?���@\�Ϳk��TQ�Bxp�                                    Bx�7�  T          @~�R?�Q�@Z�H>��@n�RBq?�Q�@I����\)���HBiQ�                                    Bx�7��  �          @z=q@\)@L�ͽ���G�B\33@\)@1G���������BLz�                                    Bx�7�P  �          @��@�\@Z�H�8Q��p�B`��@�\@<(���  �ɮBPff                                    Bx�7��  �          @��@   @aG�=�Q�?��RBq��@   @J=q������Q�Bfz�                                    Bx�7�  �          @�G�?�33@c�
���Ϳ�{Bw�R?�33@Fff��  �͙�Bj�                                    Bx�7�B  T          @�=q@�\@^{�
=q��
=Bn(�@�\@3�
�ff��BX�                                    Bx�8	�  "          @�Q�@	��@Mp��s33�]G�B`�R@	��@����Q�B@�                                    Bx�8�  �          @�
=@��@g
=    <�Bmp�@��@L�Ϳ�Q����
B`�R                                    Bx�8'4  T          @�\)@��@g�>���@�\)Bm�
@��@X�ÿ����  Bg
=                                    Bx�85�  �          @�?�@j=q?W
=A9p�B{?�@hQ�z�H�X��B~33                                    Bx�8D�  "          @�ff?�=q@j�H?aG�AB�\B~�?�=q@i���p���O33B}                                    Bx�8S&  
�          @���?��
@[��h���T  Bz=q?��
@&ff�=q���B^p�                                    Bx�8a�  "          @z�H?���@J=q���R����Bnff?���@p��%��'(�BH                                      Bx�8pr  �          @�  ?��@QG���  ��(�Bp��?��@�
�(���&33BKQ�                                    Bx�8  
�          @���?�=q@Vff�����\)Bu��?�=q@ff�-p��)=qBP�                                    Bx�8��  "          @�=q?�p�@^�R������{B~Q�?�p�@"�\�(���"p�B_
=                                    Bx�8�d  A          @�{?��
@u�
=q��(�B���?��
@H�����z�B�(�                                    Bx�8�
  g          @�33?�{@��������\)B�L�?�{@\�������B�Ǯ                                    Bx�8��  
�          @�ff?�
=@|(����� ��B�.?�
=@Z=q��p��ᙚB��\                                    Bx�8�V  �          @���?��
@�G��W
=�5�B��?��
@^{���p�B��{                                    Bx�8��  �          @���?��@��R<��
>W
=B�  ?��@n{���H��(�B��
                                    Bx�8�  �          @�(�?�p�@{�>���@�p�B�B�?�p�@g
=�˅��ffB�aH                                    Bx�8�H  �          @��R?���@z=q?��@��
B�\?���@mp���������B�#�                                    Bx�9�  g          @�Q�?�ff@g�?��A}�B�p�?�ff@mp��=p��*ffB�Q�                                    Bx�9�  �          @��?�G�@z�H���޸RB��?�G�@Y�����H���HB��                                     Bx�9 :  �          @�  ?�@��\>L��@/\)B��?�@l�Ϳ޸R��B�                                      Bx�9.�  �          @�{?�{@���>��R@�G�B��?�{@l�Ϳ�\)��  B��                                    Bx�9=�  "          @�33?���@|(�>8Q�@!�B��R?���@c�
��Q����
B��{                                    Bx�9L,  
�          @�z�?333@��>�\)@u�B��\?333@mp������HB�                                    Bx�9Z�  �          @�
=>�z�@�p�>��H@�B�(�>�z�@z=q���
��
=B���                                    Bx�9ix  �          @�Q�?5@��?
=q@��B���?5@{����R��\)B�z�                                    Bx�9x  "          @�
=?B�\@���?\(�A<��B�?B�\@}p���33�}B�ff                                    Bx�9��  �          @��?�p�@q�?.{A��B��
?�p�@j=q��
=��{B�\)                                    Bx�9�j  �          @��?\@xQ�?}p�AUB�\?\@xQ�u�N�RB�#�                                    Bx�9�  �          @��?�  @�=q?�  AV�RB�\)?�  @�녿���_33B�L�                                    Bx�9��  �          @���?E�@��
?s33AL��B��?E�@��\��{�n�\B��q                                    Bx�9�\  T          @���?^�R@��\?��\A\��B��H?^�R@��\���\�\��B��H                                    Bx�9�  s          @���?k�@���?��Alz�B�Q�?k�@����p���K�B��=                                    Bx�9ި            @���?5@���?�  A��HB�(�?5@���O\)�.�HB���                                    Bx�9�N  �          @��>�(�@���?�=qA���B��f>�(�@�ff�=p��33B�Q�                                    Bx�9��  A          @�Q�?p��@\)?���Aip�B�ff?p��@��׿s33�N{B���                                    Bx�:
�            @��H?u@���?�G�A�G�B�(�?u@��ͿL���)�B��f                                    Bx�:@  "          @�\)?:�H@��\>��@�{B��
?:�H@tz�\���\B�z�                                    Bx�:'�  
�          @���?��@��H?��@�
=B���?��@x�ÿ�z����B�B�                                    Bx�:6�  T          @�  @�@:�H�   ���HBM�\@�@��������B4G�                                    Bx�:E2  T          @{�@(��@2�\�B�\�6ffB:\)@(��@����R��(�B'��                                    Bx�:S�  
�          @�=q@n{?\>�@׮A��
@n{?��þ�������A�z�                                    Bx�:b~  "          @��H@z�H?�녾8Q���RA��@z�H?k��333���AS�
                                    Bx�:q$  "          @���@k�?Ǯ�Ǯ��Q�A��@k�?�Q쿊=q�x��A�G�                                    Bx�:�  �          @|��@P��@��u�eBQ�@P��?�(�������=qAޏ\                                    Bx�:�p  T          @tz�@/\)@%����R��B-p�@/\)@Q쿾�R��ffB��                                    Bx�:�  �          @p  @*�H@$z�������B/�R@*�H@z������33B�                                    Bx�:��  T          @n�R@"�\@)������33B8@"�\@G���{��  B'ff                                    Bx�:�b  
          @u�@"�\@3�
>\@�B?Q�@"�\@)����  �r�HB8��                                    Bx�:�  
�          @o\)@8Q�@z�>���@�=qB  @8Q�@�ͿO\)�H��B{                                    Bx�:׮  �          @qG�@�@5�>Ǯ@�Q�BHp�@�@+���  �yBB�                                    Bx�:�T  T          @s33@�R@5��W
=�Mp�BC
=@�R@�����
��
=B/�
                                    Bx�:��  �          @s33@C�
@(�>�Q�@���BQ�@C�
@ff�5�.ffB	�H                                    Bx�;�  �          @r�\@W�?�
=>�p�@��\A�@W�?�33�������A��H                                    Bx�;F  �          @{�@N{?p��?��HA�  A��H@N{?�G�?��
A�\)A�33                                    Bx�; �  �          @~�R@N{?�z�@G�A�(�A�z�@N{?�p�?�p�A��A�z�                                    Bx�;/�  �          @xQ�@C�
@z�?�=qA�=qB�@C�
@�����B�                                    Bx�;>8  
�          @e@�H@#�
?+�A+�B:��@�H@$z�!G�� ��B;�                                    Bx�;L�  �          @q�@+�@%�>L��@G�B/��@+�@���ff��=qB&                                      Bx�;[�  s          @vff@;�@��>�33@��Bp�@;�@�׿Q��H��B�                                    Bx�;j*  �          @q�@7
=@\)?n{Aep�B{@7
=@=q�u�l��B p�                                    Bx�;x�  
�          @j�H@  �#�
@7
=BOQ�C���@  ?���@!G�B/�A��
                                    Bx�;�v  
�          @h��@�?�@��B;ffAV�R@�?˅?�{B
=qBz�                                    Bx�;�  
�          @u�@Fff?0��@z�B�AIp�@Fff?˅?�  A�  A�p�                                    Bx�;��  
Z          @w�?��H?�ff����B*��?��H?���,(��S(�A��
                                    Bx�;�h  
�          @w
=?
=?�\�Vff�s=qB���?
=���
�r�\¢(�C��{                                    Bx�;�  
�          @z�H?��
?����Z=q�s  Bd�H?��
�L���p���RC�z�                                    Bx�;д  �          @~{?��\@
�H�Dz��P  B�p�?��\>��n�R�\A��                                    Bx�;�Z  �          @z�H?W
=?��Fff�`  B��
?W
=>aG��h��B�Aj�\                                    Bx�;�   �          @\)>�@0���5��6�RB���>�?���s33ǮB�k�                                    Bx�;��  T          @�=q?.{@6ff�1G��-ffB�#�?.{?����r�\��Bq                                    Bx�<L  "          @�z�?�  @'
=�=p��6Q�B��q?�  ?fff�u=qB�                                    Bx�<�  "          @�(�?��@!G��<���6�RBtQ�?��?Q��r�\ǮA�z�                                    Bx�<(�  	�          @��H?�\@)���%��=qBaG�?�\?�{�a��mB �                                    Bx�<7>  
�          @|(�@�@�����{BHp�@�?����H���WffA�{                                    Bx�<E�  
Z          @�(�@�@1G��  ��HBT(�@�?�\)�R�\�SQ�B��                                    Bx�<T�  T          @{�@	��@��{�	��BB33@	��?���G
=�P�A�33                                    Bx�<c0  
�          @z=q@�@'
=�8Q��<(�B<33@�@p���33��Q�B)=q                                    Bx�<q�  	�          @vff@(��@1G��B�\�7�B9z�@(��@ff��p����
B&��                                    Bx�<�|  
�          @q�?���@5��\)��\)B]�
?���?����#33�-��B-�
                                    Bx�<�"  T          @w
=@�@@�׿�{��z�B`=q@�@��=q�\)B9ff                                    Bx�<��  
�          @|(�@1�@0��>aG�@R�\B333@1�@"�\��{��\)B)�\                                    Bx�<�n  
�          @~�R@:�H@33?��A��HB��@:�H@*=q=���?��B)�                                    Bx�<�  
�          @k�@/\)?���@ ��B��A�Q�@/\)@
�H?�\)A�B�\                                    Bx�<ɺ  T          @hQ�@:�H?xQ�?��A��A��@:�H?�  ?���A�G�A�
=                                    Bx�<�`  
          @n�R@&ff?�{@�B�A��R@&ff@ff?ǮAȏ\B��                                    Bx�<�  y          @r�\@%�?�ff@��B��A�{@%�@�?�G�A�{B-ff                                    Bx�<��  �          @u@7
=?��?ٙ�A���B�@7
=@{?��A�B#{                                    Bx�=R  T          @u�@W
=?�G�?\)A
ffA�p�@W
=?��;aG��X��A�
=                                    Bx�=�  �          @o\)@H��?��?�A��A홚@H��?��þ\��  A�                                    Bx�=!�  "          @o\)@>{?�(�?\(�AZ=qB��@>{@	���.{�,(�B�                                    Bx�=0D  �          @:�H����?(���G��RffC�\���;��H��
�X=qCD�q                                    Bx�=>�  �          @;���p�?��\����t�HB�G���p����
�-p�¤��C6�R                                    Bx�=M�  �          @S�
��\)>�p��@  �C!�H��\)����3�
�s�\C`8R                                    Bx�=\6  �          @S�
�u?L���C33� Cff�u�=p��C�
ǮCY�H                                    Bx�=j�  �          @^{>�
=?�  �1��dG�B��H>�
=>k��Q�£W
A�33                                    Bx�=y�  �          @k�@+�@{���ͿǮB*��@+�@Q쿡G���G�B33                                    Bx�=�(  �          @l(�@Dz�?�녿8Q��3�
A���@Dz�?��ÿ\���A��R                                    Bx�=��  �          @p��@J�H?��H��G��yp�A�=q@J�H?��
��Q���33A�                                    Bx�=�t  �          @u�@e�?��H����z�A���@e�?O\)������\AK�                                    Bx�=�  �          @qG�@[�?�33�@  �<  A�@[�?&ff������=qA,��                                    Bx�=��  �          @hQ�@L��?��
������RA��@L��?!G��\��  A2�H                                    Bx�=�f  �          @k�@P��?�\)�5�4z�A�  @P��?^�R���\����Ao�                                    Bx�=�  �          @o\)@Q�?���\(��U��Aȏ\@Q�?s33��p����A���                                    Bx�=�  �          @q�@O\)?�Q�?�\@�{A܏\@O\)?�p���33��A�
=                                    Bx�=�X  �          @p��@L(�?�G�?W
=AO\)A�
=@L(�?�����Q쿱�A�33                                    Bx�>�  �          @fff@8Q�?˅=L��?Y��A�
=@8Q�?��5�FffA�ff                                    Bx�>�  �          @xQ쿁G����j=q�HC;����G����J=q�a��CqG�                                    Bx�>)J  �          @���������R�s�
33CA0�����z��Mp��QCm{                                    Bx�>7�  �          @������;��R�s33\CA������z��Mp��Q��Cl��                                    Bx�>F�  �          @��\��  �
=q�o\)�CG���  �\)�C33�B
=Cl:�                                    Bx�>U<  �          @��\���þ��
�u�CA�쿨���ff�O\)�R\)Cm�
                                    Bx�>c�  �          @���˅�L���q��C;.�˅���H�O\)�P�\Cf�3                                    Bx�>r�  �          @�33��G�=����s�
�C0LͿ�G����H�Z=q�`�HCd�                                    Bx�>�.  �          @�=q��>8Q��s33�C,녿��У��\(��g�Cd�                                    Bx�>��  �          @��ÿ�녾\)�q�p�C9����녿�33�QG��Y{Ci��                                    Bx�>�z  �          @��׿�(��B�\�n�R�C;W
��(����L���S�\Ch}q                                    Bx�>�   �          @xQ��
=�!G��\(��y�CH����
=�
�H�/\)�3�RChY�                                    Bx�>��  �          @�=q��
=��\�qG�8RCG�q��
=�{�E�ECmT{                                    Bx�>�l  �          @������
��33�tz��3CCzΰ�
�
=�Mp��QCn                                    Bx�>�  �          @�  �n{��\)�w���CD���n{��
�Q��]�Cu��                                    Bx�>�  �          @tz�
=>�{�p  �{C  �
=���H�^{#�Cx5�                                    Bx�>�^  �          @w���33>��
�u¥�fC	����33�\�b�\p�C�~�                                    Bx�?  �          @z�H�@  �����u�3C;�@  ����U�j��CxaH                                    Bx�?�  �          @~�R��
=>\�{�£u�C	�H��
=��G��i����C~J=                                    Bx�?"P  T          @w
=>�  >L���uªz�B��>�  ��\)�^�R� C�^�                                    Bx�?0�  �          @s33��;����l(� ��CPǮ����   �G��aC~��                                    Bx�??�  �          @l�Ϳ�  �h���O\)\CX(���  �z��(��*�Cq�=                                    Bx�?NB  �          @hQ��녾�
=�a�¡Q�Ca�H����G��:�H�[G�C�Q�                                    Bx�?\�  �          @a녽#�
�#�
�^{²8RCaJ=�#�
��z��C33�uC�Y�                                    Bx�?k�  �          @O\)�=p��\)�FffCY�=p���Q��{�H  Cy33                                    Bx�?z4  �          @`  �\�\�[�¢33Ca� �\��Q��6ff�\�C��                                    Bx�?��  �          @^{<��
�#�
�]p�®�=C��
<��
��G��>�R�m��C�U�                                    Bx�?��  �          @`  >��8Q��\(�¢��C���>���\�=p��h�C��                                    Bx�?�&  �          @aG�>\=L���_\)§� A ��>\��=q�G
=�yp�C���                                    Bx�?��  �          @a�?!G�>��H�Z�H�
B�?!G����P  ��C�%                                    Bx�?�r  �          @[�?^�R?\)�N�RaHBp�?^�R��G��G���C�c�                                    Bx�?�  �          @>{@��?�(�>\@�B
=@��?ٙ�����B�R                                    Bx�?�  �          @HQ�@&ff?�Q�=�@  B=q@&ff?�ff�.{�K
=A��H                                    Bx�?�d  �          @?\)@ff?��
�����G�A�p�@ff>�
=��33���
A#�                                    Bx�?�
  �          @B�\@ff?^�R�W
=��=qA��@ff?+��
=�Yp�A}�                                    Bx�@�  �          @tz�@(��?c�
@&ffB,ffA�
=@(��@   ?��A�B�R                                    Bx�@V  �          @s�
@$z�?(��@-p�B6G�Af�\@$z�?���@B�B�R                                    Bx�@)�  �          @y��@5�?�
=@��Bz�A���@5�@	��?ǮA��B��                                    Bx�@8�  �          @z�H@5?�  @   B  A�@5@33?�  A�=qB�                                    Bx�@GH  �          @xQ�@3�
?&ff@%B'�AQ��@3�
?��
?�p�A��\B�                                    Bx�@U�  �          @w�@2�\?�@'
=B*=qA,��@2�\?�Q�@�
BQ�A��H                                    Bx�@d�  �          @u@5?�@!�B$��A%�@5?У�@   A�G�A�G�                                    Bx�@s:  �          @tz�@   ?c�
@.{B6�HA�p�@   @�
@   A�33B��                                    Bx�@��  �          @p��@p�?�{@'
=B/�A��@p�@(�?��A�ffB&�                                    Bx�@��  �          @r�\@)��?���@ffB�
A��@)��@��?�(�A�
=B!�H                                    Bx�@�,  �          @o\)@�R?��@�B#  A���@�R@33?ǮA�\)B(�                                    Bx�@��  �          @W
=@4z�?}p�?5AV�HA��R@4z�?��H>.{@N�RA��                                    Bx�@�x  �          @_\)@H��?z�H��z����RA��R@H��>��ÿ�p��ȣ�@�=q                                    Bx�@�  �          @aG�@R�\?#�
��ff��
=A0��@R�\=��
��p�����?���                                    Bx�@��  �          @c33@X��>aG��Tz��[\)@n�R@X�þB�\�W
=�]��C�j=                                    Bx�@�j  �          @dz�@`  ��Q�&ff�(Q�C�AH@`  �\�
=q�
{C���                                    Bx�@�  �          @e�@a�=L�;�G����H?\(�@a녾#�
�����C��)                                    Bx�A�  �          @e@a�>�׾�z���@�\@a�>�\)������@�=q                                    Bx�A\  �          @c�
@^{?+����R��G�A/
=@^{>�ff�
=�(�@���                                    Bx�A#  �          @c�
@^{?zᾞ�R���HA\)@^{>�p�����{@��                                    Bx�A1�  �          @e@^�R?&ff���
=A*ff@^�R>��
�O\)�P��@�{                                    Bx�A@N  �          @k�@e>�p��z���\@��@e=u�0���-p�?z�H                                    Bx�AN�  �          @l(�@g�>�������\@�\)@g�<������R>��                                    Bx�A]�  �          @l��@h��>�
=��ff��ff@�  @h��>#�
�
=��\@#�
                                    Bx�Al@  �          @g�@b�\?.{���Ϳ��A.=q@b�\?\)�����˅A�
                                    Bx�Az�  �          @k�@S33?��þ�\)��33A�@S33?��\�xQ��up�A�(�                                    Bx�A��  �          @qG�@W
=?ٙ������A���@W
=?����c�
�[33A�G�                                    Bx�A�2  �          @s�
@U�?���>L��@<��A�@U�?��H�(��� ��Aم                                    Bx�A��  �          @w
=@O\)@   ?�\@�\A��@O\)@ �׾����  A�ff                                    Bx�A�~  �          @y��@C33@33?8Q�A*�HB(�@C33@Q��
=��z�B��                                    Bx�A�$  �          @x��@B�\@33?E�A6�HBz�@B�\@�þ�p����RB{                                    Bx�A��  �          @u@\(�?��H?G�A=��A�  @\(�?�z�<�>�
=A�                                    Bx�A�p  �          @r�\@W
=?�p�?h��A]A�{@W
=?�p�>�?�
=A��                                    Bx�A�  �          @tz�@U�?��
?��A}A�\)@U�?�>k�@X��A���                                    Bx�A��  T          @u�@dz�?s33?�  AqAn�R@dz�?�ff>�ff@���A�Q�                                    Bx�Bb  �          @vff@j=q?L��?Tz�AG�AD��@j=q?��>�p�@�=qA�z�                                    Bx�B  �          @z�H@j�H?8Q�?�33A���A2�H@j�H?�?0��A"�RA�p�                                    Bx�B*�  �          @j�H@W�?Q�?�{A�{A[�@W�?��R?(�A
=A���                                    Bx�B9T  �          @mp�@^�R>�(�?n{An�H@��@^�R?O\)?#�
A!��AQ�                                    Bx�BG�  �          @e@O\)�u?�p�AîC�}q@O\)?�R?��A���A-p�                                    Bx�BV�  �          @`��@N�R�G�?k�AxQ�C�33@N�R��\)?�
=A��C���                                    Bx�BeF  �          @Vff@:=q��
=?:�HAL(�C��@:=q�p��?�ffA��RC��                                    Bx�Bs�  �          @O\)@1녿��
>�{@�ffC��
@1녿���?�  A�z�C�Ff                                    Bx�B��  �          @J�H@ ����
��\)��(�C�R@ ���ff�u��{C�B�                                    Bx�B�8  �          @Vff@
=q��@
=B;��C��@
=q?(��@�
B6A�                                    Bx�B��  �          @O\)?�ff��@6ffB�ǮC�>�?�ff?�
=@%B_�B)(�                                    Bx�B��  T          @L��?�z�=�G�@8��B�
=@��?�z�?�z�@!G�BXG�BJp�                                    Bx�B�*  �          @G
=?n{�8Q�@1G�B��C�Ff?n{?�R@2�\B�L�B�R                                    Bx�B��  �          @(Q�>�ff���
@�
B�(�C���>�ff>�@!G�B�
=A�                                      Bx�B�v  �          @(Q�>�=q��R@{B�G�C���>�=q?\)@\)B���B�aH                                    Bx�B�  �          @@  ?У׾�G�?ٙ�B4\)C�h�?У�>�p�?�(�B6��AM��                                    Bx�B��  �          @G
=@+��.{?�
=A��
C�,�@+�>��H?���A��HA&�\                                    Bx�Ch  �          @Fff@8Q�>u>��HA�H@���@8Q�>�G�>���@ϮA
{                                    Bx�C  �          @G
=@'�?�33��\)��(�A��
@'�?�  �
=�<��A�Q�                                    Bx�C#�  �          @u@mp�?8Q�>��@w�A/33@mp�?B�\���Ϳ���A8Q�                                    Bx�C2Z  �          @x��@u>Ǯ�8Q��+�@��@u>�=q��{����@���                                    Bx�CA   �          @u�@e?�  >��@�p�A��@e?�ff�L���;�A��                                    Bx�CO�  �          @z�H@fff?���?J=qA9��A��@fff?��
=�G�?���A�z�                                    Bx�C^L  �          @{�@j=q?��?(�A=qA��@j=q?��#�
�
=qA�                                      Bx�Cl�  �          @�Q�@l(�?�
=?#�
A�\A�G�@l(�?Ǯ��\)���\A���                                    Bx�C{�  �          @\)@i��?��
?   @�RA�@i��?˅�u�[�A�ff                                    Bx�C�>  �          @���@k�?�p�?\)A ��A�G�@k�?��þ#�
��A�G�                                    Bx�C��  �          @�33@e�?�\)?�z�A��HA�G�@e�?���?#�
A��A���                                    Bx�C��  �          @���@e�?��?��A�p�A��\@e�@   ?\)@�
=A��
                                    Bx�C�0  �          @�z�@a�?�  ?�(�A���A�\)@a�?��?z�HA[�A�=q                                    Bx�C��  �          @�p�@g
=?�{?�\A�p�A�33@g
=?��?��ArffA���                                    Bx�C�|  �          @��
@g
=?�Q�?���A��A�(�@g
=?�\?^�RAC\)A�z�                                    Bx�C�"  �          @�z�@e?�Q�?�
=A�  A�G�@e?���?xQ�AYA�33                                    Bx�C��  �          @��\@c33?�ff?\A�G�A��R@c33?�?G�A1�A��                                    Bx�C�n  �          @��H@hQ�?��H?�A��A�p�@hQ�?�(�?=p�A%�A�z�                                    Bx�D  �          @���@k�?�  ?�A��HA��@k�?�{>\@�
=A�
=                                    Bx�D�  �          @�p�@o\)?��?�=qAo�A��@o\)?�{>�\)@x��Aә�                                    Bx�D+`  �          @���@Vff?�  ?��A��A�\)@Vff@z�?L��A6�HA��                                    Bx�D:  �          @�=q@U�?��?�  A���A�R@U�@G�?
=q@�ffB	Q�                                    Bx�DH�  �          @���@Z=q?�Q�?�{A���A�\@Z=q@>�Q�@�
=B	�                                    Bx�DWR  �          @��@U�@33?�z�A�A���@U�@ff>�?��
B�                                    Bx�De�  �          @�=q@O\)@G�?^�RAF�HB
=@O\)@�H�aG��H��B{                                    Bx�Dt�  �          @�(�@9��?�\)@
=qB  A�  @9��@=q?�(�A�33B��                                    Bx�D�D  �          @���@#�
?s33@G
=BB�HA�(�@#�
@{@�B�B#                                    Bx�D��  �          @�
=@)��?�33@=p�B2�A��@)��@$z�@�\A�\)B0=q                                    Bx�D��  �          @��@1�?�z�@'
=B�A�z�@1�@(��?�\)A���B.{                                    Bx�D�6  �          @�(�@#33@ ��@!�B�RBQ�@#33@:�H?��A�\)BC�                                    Bx�D��  �          @�G�@	��@�R@B
=BDz�@	��@P��?�G�Ahz�Bbp�                                    Bx�D̂  �          @���@
�H@=q@Q�B33B@33@
�H@Mp�?�=qAy�B_��                                    Bx�D�(  
�          @���@
�H@33@{B�HB:�
@
�H@I��?��HA�{B]�H                                    Bx�D��  �          @�  @�@�\@*�HB'{B,Q�@�@@  ?�  A�Q�BX
=                                    Bx�D�t  �          @�=q@z�@
=@3�
B.ffB6(�@z�@G�?���A�\)Bb
=                                    Bx�E  �          @�33@{@	��@+�B#�HB0�@{@G
=?�(�A�z�BY��                                    Bx�E�  �          @�(�@%�@�H@�A�\)B,Q�@%�@Fff?Y��A>{BH                                    Bx�E$f  �          @�@   @
�H@$z�BffB#�
@   @Dz�?�{A�(�BK��                                    Bx�E3  �          @��@   @   @)��Bz�B  @   @=p�?�G�A�z�BGff                                    Bx�EA�  �          @��
@��?�\)@,��B$�RB�\@��@7
=?�{A�\)BE�R                                    Bx�EPX  �          @�ff@/\)?��H@4z�B)  A�  @/\)@"�\?�33A�  B+G�                                    Bx�E^�  �          @�33@\)?�\)@7�B0  B  @\)@,��?��A؏\B=ff                                    Bx�Em�  �          @��@P  @"�\��
=��33B
=@P  @ff��p����
B�                                    Bx�E|J  �          @��@Tz�@33�333���B
�H@Tz�?�\�У���33A�ff                                    Bx�E��  �          @�z�@`��@�����(�A�z�@`��?����xQ��YA�Q�                                    Bx�E��  �          @��H@e�?�Q�>���@�ffA��@e�?�\)��� Q�A���                                    Bx�E�<  �          @���@g�?��H?xQ�A_�A�{@g�?޸R>u@^{A�\)                                    Bx�E��  �          @�(�@h��?�\?:�HA$  A��@h��?���G����RA�(�                                    Bx�Eň  �          @��@Z�H@
�H?�A   B\)@Z�H@�;�
=��p�B��                                    Bx�E�.  �          @�(�@dz�?�Q�?0��A33A�@dz�@33�W
=�9��A�\                                    Bx�E��  �          @��\@c33?�{?=p�A'�
Aܣ�@c33?��R������A�z�                                    Bx�E�z  �          @��H@e?�=q?(�A��A�  @e?�z�u�XQ�A�Q�                                    Bx�F    �          @��@U@�?�RAG�B��@U@\)�\���
B33                                    Bx�F�  �          @��@[�@�?8Q�A#
=A�\@[�@�þaG��J=qA�p�                                    Bx�Fl  �          @�(�@W�@z�>��H@�Q�B
�@W�@�
�\)��Q�B	\)                                    Bx�F,  �          @�G�@O\)@�>�ff@�{B�H@O\)@33�
=�\)Bff                                    Bx�F:�  �          @���@Dz�@!G�?��A	Bz�@Dz�@"�\�
=q����B(�                                    Bx�FI^  �          @���@N�R@ff?�A�B��@N�R@
=���H��ffB�
                                    Bx�FX  �          @�=q@fff?���>���@�{A�G�@fff?�ff�����
A�(�                                    Bx�Ff�  �          @��@a�?��H>#�
@{A��@a�?��333��\A�z�                                    Bx�FuP  �          @��@L��@p�>�  @a�Bp�@L��@�Q��<(�B�\                                    Bx�F��  �          @��\@a�?�
=>��@�p�A�@a�?����������
A�\)                                    Bx�F��  �          @�=q@mp�?�33=�?�Q�A��@mp�?���
=�{A��R                                    Bx�F�B  �          @�Q�@W
=?�?�z�A�ffA�p�@W
=@	��>�=q@tz�B�R                                    Bx�F��  �          @�  @h��?�G�?.{Az�A�z�@h��?�z����Q�A��
                                    Bx�F��  �          @�Q�@dz�?�\=�Q�?�ffA�z�@dz�?�녿+���A��                                    Bx�F�4  �          @���@i��?��;������A���@i��?��ÿxQ��`��A�ff                                    Bx�F��  �          @�G�@j�H?�>B�\@,(�A�p�@j�H?˅����\A��                                    Bx�F�  �          @�G�@hQ�?޸R>�=q@q�A���@hQ�?�Q���H���AǙ�                                    Bx�F�&  T          @�@q�?ٙ�?�\@�Q�A�G�@q�?�  ��  �^{A�
=                                    Bx�G�  �          @�z�@l��?�
=?=p�A%��Aî@l��?�����AӅ                                    Bx�Gr  �          @�33@b�\?��?�  Ab�RA�Q�@b�\@�\>#�
@��A��                                    Bx�G%  �          @��@a�?�z�?h��AL(�A�33@a�@
=<��
>���A�\)                                    Bx�G3�  �          @��@Y��@��?k�AMp�B�
@Y��@�ý�Q쿗
=B\)                                    Bx�GBd  �          @�p�@c�
?��H?@  A'33A�ff@c�
@�\)��
=A�p�                                    Bx�GQ
  �          @�Q�@~{?�  ��33��{A�p�@~{?��H�p���L��A�(�                                    Bx�G_�  �          @��R@~{?�{�aG��>�RA��@~{?�녿E��((�A�z�                                    Bx�GnV  �          @�
=@{�?��R����ffA��@{�?���&ff��\A�z�                                    Bx�G|�  �          @�  @|��?�ff���
��=qA��@|��?��׿8Q��(�A�                                    Bx�G��  �          @�Q�@x��?�p�<�>�A��@x��?�=q�0�����A�p�                                    Bx�G�H  �          @��R@w�?��������A���@w�?�������k
=A��                                    Bx�G��  �          @��R@w
=?Ǯ�!G��
�\A�=q@w
=?�z῜(���G�A���                                    Bx�G��  T          @�@q�?�녿(����A�@q�?�(����
���A�
=                                    Bx�G�:  �          @�(�@n{?�녿5��\A�(�@n{?�����=q��A��R                                    Bx�G��  �          @�p�@o\)?�p��z���{AƸR@o\)?�����R����A��                                    Bx�G�  �          @���@n{?��H�&ff�  A�
=@n{?����ff���A��\                                    Bx�G�,  �          @�p�@tz�?��
�5�  A��\@tz�?��Ϳ��\��(�A��R                                    Bx�H �  �          @���@qG�?�
=��\�߮A�  @qG�?��ÿ�33���RA��\                                    Bx�Hx  �          @�{@xQ�?�(��
=��A�(�@xQ�?��Ϳ���|��A}�                                    Bx�H  �          @��@tz�?�G�����ffA�ff@tz�?�녿����
A���                                    Bx�H,�  �          @�{@u?�녾�G���Q�A��@u?��ÿ����l��A�\)                                    Bx�H;j  �          @�\)@z=q?��þ�����(�A���@z=q?��\��  �[�A��                                    Bx�HJ  T          @��@{�?��þL���0��A�@{�?���Tz��4  A�33                                    Bx�HX�  �          @�ff@z�H?��R��  �[�A�
=@z�H?�G��W
=�8z�A���                                    Bx�Hg\  �          @��R@xQ�?��þ�{��ffA�Q�@xQ�?���u�R=qA��                                    Bx�Hv  �          @�@u?�  �(����A��@u?�{���H��33A�z�                                    Bx�H��  �          @���@q�?�=q�!G��33A�G�@q�?�Q쿚�H��Q�A��                                    Bx�H�N  �          @�{@y��?�녿#�
���A��R@y��?�G���33�~�RAg�
                                    Bx�H��  �          @�\)@}p�?���5�{A�{@}p�?h�ÿ�
=��33AN�\                                    Bx�H��  �          @�p�@x��?�ff�8Q���
A��@x��?fff��Q����\AO�
                                    Bx�H�@  �          @���@u?�
=�n{�O�A���@u?5��=q���A'33                                    Bx�H��  �          @��@s�
?�녿���\)A��@s�
?녿��
��A	G�                                    Bx�H܌  T          @��
@j�H?@  �ٙ���A:{@j�H    ��{�ׅ=u                                    Bx�H�2  �          @�@fff?�=q��=q�υA��@fff>�  �
=��ff@���                                    Bx�H��  �          @�p�@e?}p���{��ffAv�\@e>#�
�ff��\@!�                                    Bx�I~  
�          @��@\(�?c�
�
=q��Q�Ah(�@\(��L������C��3                                    Bx�I$  �          @�ff@^�R?8Q���R� ffA:�H@^�R�k����
=C�3                                    Bx�I%�  �          @�
=@h��?W
=���H��
=AO
=@h�ü��
�Q���Q�C��                                     Bx�I4p  �          @��@k�?^�R��33��
=AS�@k�=#�
����{?��                                    Bx�IC  �          @�Q�@l(�?��\������
Aw33@l(�>8Q��Q���Q�@5                                    Bx�IQ�  �          @���@l(�?�33��=q��=qA�(�@l(�>����Q���G�@���                                    Bx�I`b  �          @��@i��?�z����ģ�A�
=@i��?z������Q�A��                                    Bx�Io  �          @�=q@fff?�Q��Q���{A���@fff?aG��{��p�A[�                                    Bx�I}�  �          @���@e�?\��  ���
A�ff@e�?333�p���A2{                                    Bx�I�T  �          @�=q@n�R?�p���\��{A�  @n�R>�(��
=���H@љ�                                    Bx�I��  �          @��@~�R?�\>u@I��A�p�@~�R?�(����ə�A��R                                    Bx�I��  �          @��
@�  ?У�?5AA��@�  ?��
<#�
>�A��                                    Bx�I�F  �          @��
@���?��
?E�A"�HA�
=@���?��H=�G�?���A��R                                    Bx�I��  �          @�=q@{�?ٙ�>��R@��\A���@{�?�Q�Ǯ���RA��                                    Bx�IՒ  �          @��H@�  ?�\)>u@J=qA�Q�@�  ?�=q��
=��G�A��R                                    Bx�I�8  �          @��H@�G�?��H?�@��HA���@�G�?��ýL�Ϳ��A��                                    Bx�I��  �          @��@�33?��>���@�G�A�=q@�33?�=q�k��G�A��                                    Bx�J�  �          @�=q@���?��������33A�z�@���?�=q�c�
�@(�Anff                                    Bx�J*  �          @���@x��?���ff�c
=A�z�@x��?fff�\���\APQ�                                    Bx�J�  �          @�\)@l(�?�Q쿺�H��p�A�(�@l(�?B�\��33��(�A9G�                                    Bx�J-v  �          @��@e�?�\)��Q���=qA�
=@e�>�\)�{����@�
=                                    Bx�J<  �          @��@`  ?�=q�G���(�A�33@`  =�G�� ����?�\)                                    Bx�JJ�  �          @��H@p��?��ÿ������A33@p��>���
=q���H@x��                                    Bx�JYh  �          @���@o\)?��H��Q���A��R@o\)?J=q����ҸRA>�H                                    Bx�Jh  �          @���@p��?�G���=q�ɅAo�
@p��>aG������@W
=                                    Bx�Jv�  �          @��H@|(�?�=q��Q���G�At��@|(�>�G���  ��  @˅                                    Bx�J�Z  T          @��\@|��?�����
��Q�Av�R@|��?�\�������R@��                                    Bx�J�   �          @��@��?�
=�W
=�3�
A�\)@��?B�\��p���z�A*ff                                    Bx�J��  �          @��\@��?��s33�K�A�ff@��?5��=q����A33                                    Bx�J�L  �          @��
@�33?�G��Y���2�HA���@�33?W
=��G�����A8��                                    Bx�J��  �          @���@��
?��\�aG��9p�A�
=@��
?Tzῦff��A733                                    Bx�JΘ  �          @�p�@���?���Tz��-A�z�@���?aG���G�����A>�H                                    Bx�J�>  �          @��@���?�G��Tz��-��A�\)@���?Y����  ���A9�                                    Bx�J��  �          @��H@�33?�(��8Q���A��R@�33?Y�������r{A<z�                                    Bx�J��  �          @�G�@��\?�녿&ff�
�HAy@��\?O\)���
�^�HA3\)                                    Bx�K	0  �          @�{@l��?�{����|Q�A�Q�@l��?����z���G�A�                                    Bx�K�  �          @��R@l��?�Q쿈���j�RAĸR@l��?�z�У���{A���                                    Bx�K&|  �          @�
=@j�H?�����\�`Q�A�\)@j�H?��\��\)����A�=q                                    Bx�K5"  �          @�
=@q�?�Q�J=q�,z�A��H@q�?��\��\)���A�z�                                    Bx�KC�  
�          @�
=@s33?�녿Q��3�A�z�@s33?��H������z�A���                                    Bx�KRn  �          @�G�@��
?���   ����Af�H@��
?L�Ϳ\(��9�A.�H                                    Bx�Ka  �          @�G�@�(�?n{�!G���RAJ�R@�(�?!G��n{�H��A	                                    Bx�Ko�  �          @���@�(�?J=q�L���,(�A,��@�(�>�(�����`(�@�ff                                    Bx�K~`  �          @���@�(�?aG��O\)�.=qAA�@�(�?�\��=q�ip�@�                                    Bx�K�  �          @���@��
?Q녿k��F=qA3�@��
>�녿�z��z�R@�\)                                    Bx�K��  �          @�G�@�33?!G���=q�iG�Az�@�33>8Q쿞�R��=q@%�                                    Bx�K�R  �          @�G�@��\?(���33�xQ�A\)@��\>\)�����(�?�                                    Bx�K��  �          @���@�=q?�\��������@�ff@�=q<������ff>�(�                                    Bx�KǞ  �          @�G�@��
>�
=����u�@�G�@��
�����H���C���                                    Bx�K�D  �          @���@���?z῜(���33A��@���=��
������p�?�                                    Bx�K��  �          @���@���?J=q��(���z�A1@���>�\)�����@���                                    Bx�K�  �          @�  @{�?c�
������Q�AMp�@{�>��ÿ˅��\)@��                                    Bx�L6  �          @�
=@u�?��������A}��@u�?   ��Q���G�@�                                    Bx�L�  �          @��@r�\?���p����
A��@r�\?�Ϳ���Q�A
=                                    Bx�L�  �          @���@r�\?�p��������A�33@r�\?녿�z��ӅA�                                    Bx�L.(  �          @���@|(�?xQ쿷
=���A\��@|(�>\��Q�����@�{                                    Bx�L<�  �          @���@�Q�?:�H��ff��A$Q�@�Q�>L�Ϳ�p����@7
=                                    Bx�LKt  �          @�Q�@{�?W
=��33���A@��@{�>�=q��{���@|��                                    Bx�LZ  �          @�  @tz�?�G���=q��(�Al��@tz�>�p����̣�@�
=                                    Bx�Lh�  �          @�ff@mp�?��
����
=Ax  @mp�>�33��
=��z�@�{                                    Bx�Lwf  �          @�{@o\)?aG���z���G�AS\)@o\)>W
=��\)��  @Q�                                    Bx�L�  �          @��R@p  ?^�R�ٙ���(�AP��@p  >L�Ϳ�33��=q@?\)                                    Bx�L��  �          @�
=@o\)?Q녿����z�AE@o\)=����H�ݮ?��                                    Bx�L�X  �          @��@j=q?fff��\��33A\��@j=q>L�Ϳ�(���z�@HQ�                                    Bx�L��  �          @�z�@h��?z�H��p�����Ap��@h��>�z���H��(�@�=q                                    Bx�L��  �          @��@n�R?L�Ϳ�z����AAp�@n�R>����=q����@z�                                    Bx�L�J  �          @�p�@j=q?Y�����ͮARff@j=q>�����R���@�                                    Bx�L��  �          @�z�@l��?
=��G���G�Az�@l�ͽ��
�������HC�`                                     Bx�L�  �          @��@k�?G���z���ffA@(�@k�>\)��=q��\)@
=q                                    Bx�L�<  �          @�@o\)?
=��G���p�A  @o\)��\)������
=C�n                                    Bx�M	�  �          @�ff@p��?5��p���=qA*�H@p��=L�Ϳ�{���?=p�                                    Bx�M�  �          @�(�@k�?
=��\��G�A@k����
��\)����C�ff                                    Bx�M'.  �          @�{@k�?z��z���z�A�R@k������R��\C��{                                    Bx�M5�  �          @�\)@s33?!G���  ���A��@s33�#�
��{����C��)                                    Bx�MDz  �          @�p�@p��>���p����
@�G�@p�׾#�
�������C���                                    Bx�MS   T          @�ff@w
=>�ff�˅��  @�@w
=�\)��33����C��)                                    Bx�Ma�  �          @�p�@s33>�33��
=���R@��@s33����ٙ�����C�f                                    Bx�Mpl  �          @���@fff>�p������@���@fff��Q�������C��                                    Bx�M  �          @��H@\(�>�p��(��p�@�{@\(���
=��� �HC���                                    Bx�M��  �          @�(�@^{>�
=���� ��@ۅ@^{�\�p��G�C���                                    Bx�M�^  �          @��
@\(�>����\)��@���@\(��   ����=qC��                                    Bx�M�  �          @�(�@]p�>�  ��R��@�@]p��
=q�����C���                                    Bx�M��  �          @��@j=q>�����H���\@�  @j=q��G�����ffC��3                                    Bx�M�P  �          @�p�@l��>����33��p�@Q�@l�Ϳ�����p�C�
=                                    Bx�M��  �          @���@qG�>��
��Q���z�@��H@qG���z�ٙ���33C��
                                    Bx�M�  �          @�z�@q�>W
=��z����@G�@q녾\�У���C�!H                                    Bx�M�B  �          @��\@tz�=�\)�����  ?��@tz��녿�=q����C��                                    Bx�N�  �          @�=q@hQ콸Q���ָRC�Ff@hQ�8Q��Q�����C�g�                                    Bx�N�  
�          @���@a녽��
���R��C�W
@a녿B�\���أ�C��=                                    Bx�N 4  �          @��@\(��k������ffC�@\(��s33��
=��
=C�C�                                    Bx�N.�  �          @��H@\(��.{����
=C��)@\(��h��� ����G�C��q                                    Bx�N=�  �          @��@W��\)�  ���C�Ǯ@W��fff�z���\)C�~�                                    Bx�NL&  �          @���@U��.{���	�\C���@U��p������RC��                                    Bx�NZ�  �          @�Q�@N{�������C���@N{�s33����C�˅                                    Bx�Nir  �          @z�H@E>L��������@o\)@E��R��
�G�C�T{                                    Bx�Nx  �          @�Q�@O\)=�Q��ff��H?�ff@O\)�5�\)�G�C�˅                                    Bx�N��  �          @{�@O\)�#�
�{�	�HC���@O\)�J=q������C�'�                                    Bx�N�d  �          @w
=@E�>.{��
�=q@Dz�@E���R��R�ffC�J=                                    Bx�N�
  �          @{�@>�R=�Q��"�\�"
=?�Q�@>�R�E����{C���                                    Bx�N��  �          @x��@G��u��\���C��3@G��}p������C�/\                                    Bx�N�V  �          @z=q@H�ü#�
�z���HC�޸@H�ÿJ=q����C��                                    Bx�N��  t          @z�H@Mp��.{�\)�Q�C�w
@Mp��h���33����C�R                                    Bx�Nޢ  �          @z�H@A녾u�p���C��f@A녿���\)��
C���                                    Bx�N�H  �          @u@:=q�����{� �\C�Ǯ@:=q�����{���C�aH                                    Bx�N��  �          @u�@7
=�k��!G��%{C���@7
=��ff�33�(�C��                                    Bx�O
�  �          @s33@1G��+�� ���%=qC�(�@1G���(�����	\)C�\                                    Bx�O:  �          @w�@>�R>�\)�(����@�@>�R�
=q������C��)                                    Bx�O'�  �          @s�
@B�\?��׿����A��@B�\>�ff�
=q��A�                                    Bx�O6�  �          @u�@.�R?�33�=q�\)A�
=@.�R>����)���0
=@�{                                    Bx�OE,  �          @w�@%?=p��0  �6��A~{@%�W
=�5�>\)C��{                                    Bx�OS�  �          @~�R@X��?�G����\�n=qAۅ@X��?�=q�������A�p�                                    Bx�Obx  �          @|��@L(�?�  ���
��(�A�\@L(�?�z��G����A�                                    Bx�Oq  �          @|(�@G�?�=q��G����A�R@G�?�p�����
=A�z�                                    Bx�O�  �          @z=q@*=q?����p��	�\B	��@*=q?�  �+��-33A�p�                                    Bx�O�j  t          @|(�@7
=?�(����߅B
\)@7
=?��\�Q��33A�33                                    Bx�O�  �          @w
=@0  ?����Q���RA�33@0  ?J=q�!G��%�A�z�                                    Bx�O��  �          @p  ?��R@E��G����B�k�?��R@(����!\)Bj                                      Bx�O�\  �          @mp�?���@HQ������
=B��f?���@p��   �)��Bp�                                    Bx�O�  �          @qG�?��@E��
=��B�?��@���%��-G�Br�                                    Bx�Oר  �          @p  ?��\@AG�������ffB��?��\@��,���7��Bs�                                    Bx�O�N  �          @p  ?�33@B�\��\)�B���?�33@��0  �<ffB}33                                    Bx�O��  �          @p��?��R@@�׿�{��(�B�.?��R@���.�R�:�Bt                                    Bx�P�  �          @r�\?�
=@C33�����ffB���?�
=@�\�1G��<33Bz�                                    Bx�P@  �          @tz�?��R@>�R��� p�B���?��R@��7��CQ�Bq�H                                    Bx�P �  �          @u�?�
=@:�H����BQ�?�
=@��6ff�@33B_�H                                    Bx�P/�  �          @xQ�?�z�@=p��{��
B�(�?�z�@ff�B�\�N�Btp�                                    Bx�P>2  �          @xQ�?��@7��G��Q�B���?��@ ���Dz��Pp�Be�                                    Bx�PL�  T          @u?�\)@"�\����Be��?�\)?��C33�R��B7
=                                    Bx�P[~  �          @tz�?�{@{�(����Bc��?�{?����E�W�
B1�                                    Bx�Pj$  �          @x��?˅@
=�)���+B`(�?˅?�z��P���cp�B&G�                                    Bx�Px�  �          @x��?�(�@G��5�;BF{?�(�?��
�U��k�A���                                    Bx�P�p  �          @w�?޸R?���8���Az�B=G�?޸R?c�
�U��np�A���                                    Bx�P�  �          @w
=?�  ?��5�>
=B>�H?�  ?p���S33�k�A�R                                    Bx�P��  �          @{�?��@0���   �z�B��?��?��N�R�\��B\G�                                    Bx�P�b  �          @~{?�  @(Q��$z��!z�Bq��?�  ?ٙ��P���\�
BB�                                    Bx�P�  �          @z=q?��@z��.�R�2  Ba33?��?�{�Tz��h�HB%�                                    Bx�PЮ  �          @qG�?��R@��0  �:B[G�?��R?��QG��o��B                                      Bx�P�T  �          @p  ?Ǯ@  �!��+
=B]�?Ǯ?�{�Fff�a�B$z�                                    Bx�P��  �          @w�@�?����1��7��A���@�>aG��?\)�J�@���                                    Bx�P��  T          @~�R@&ff?�33�2�\�1�A�\)@&ff>u�@���DQ�@��                                    Bx�QF  T          @xQ�@*=q?�z��$z��&33A��
@*=q>��R�333�9{@׮                                    Bx�Q�  �          @y��@)��?��\�#�
�$
=A̸R@)��>�(��4z��9�A33                                    Bx�Q(�  �          @|(�@-p�?�  �$z��"��A�Q�@-p�>���5��7�A	��                                    Bx�Q78  �          @�=q@333?�=q�0  �)�RA��
@333>B�\�<(��9=q@y��                                    Bx�QE�  �          @��@:=q?W
=�5��,G�A���@:=q�u�=p��5�C�q�                                    Bx�QT�  �          @��
@C�
?(��,(��#{A4��@C�
�k��0  �'p�C�ٚ                                    Bx�Qc*  �          @��H@@  >�ff�0  �(�A�@@  �����0  �)33C�.                                    Bx�Qq�  �          @�z�@8Q�>����<(��533@���@8Q�   �:�H�3��C��                                    Bx�Q�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Q��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�Q�h   D          @���@5=�G��L���Az�@p�@5�Y���E�8��C��\                                   Bx�Q�  �          @�p�@.�R�#�
�I���D(�C��H@.�R�s33�@  �8G�C�s3                                   Bx�Qɴ  �          @�  @5�    �J�H�A  <��
@5��p���A��5��C���                                   Bx�Q�Z  �          @��
@.�R�#�
�Dz��A(�C��=@.�R�s33�:�H�4��C�l�                                   Bx�Q�   �          @�ff@&ff�L���R�\�N(�C�ٚ@&ff��z��E��=
=C�f                                   Bx�Q��  �          @�@*�H��33�J�H�FC�Ff@*�H���\�;��3
=C�L�                                   Bx�RL  �          @���@*=q�5�=p��<G�C�~�@*=q��ff�'��!��C��                                     Bx�R�  �          @��@'��u�J�H�I{C�XR@'���z��=p��7�\C�\                                    Bx�R!�  t          @�@(Q쾊=q�N�R�J��C�f@(Q쿚�H�@���8Q�C��3                                    Bx�R0>  �          @�@#33�&ff�N�R�K(�C���@#33�Ǯ�9���0Q�C�B�                                    Bx�R>�  �          @�@$z῵�>{�5�\C��\@$z��{����C��{                                    Bx�RM�  T          @���@�R����&ff��C�o\@�R�#33��Q���p�C�
                                    Bx�R\0  
�          @���@
=��=q�,(��'�HC�!H@
=�!���\��z�C���                                    Bx�Rj�  4          @}p�@���Q��.{�-33C��@�����
=� �C�\                                    Bx�Ry|  �          @}p�@������,(��+��C���@����Q��p�C�l�                                    Bx�R�"  t          @}p�@!녿�  �8Q��:Q�C�*=@!녿���{��C�]q                                    Bx�R��  
�          @u@(��k��@���N�RC���@(��޸R�'��,Q�C�                                    Bx�R�n  �          @w�@{��ff�G��X\)C�=q@{��=q�7��?�
C��\                                    Bx�R�  �          @z�H@Q�xQ��Fff�S��C���@Q�����,(��/�C�Ǯ                                    Bx�Rº  �          @�G�?�p��>�R�{���C�8R?�p��c33�\���
C���                                    Bx�R�`  �          @y��?L���QG������  C��f?L���j�H�aG��S33C�+�                                    Bx�R�  �          @tz�?0���3�
�
=��C���?0���W
=��(����C��\                                    Bx�R�  �          @vff?������:�H�EC�Ǯ?����1G���R��C��3                                    Bx�R�R  �          @w
=?��
���8���B��C�C�?��
�#33����  C�xR                                    Bx�S�  �          @q�?�\)��
=�;��L�\C��
?�\)�*�H�����C��3                                    Bx�S�  �          @q�?�z��(��8���H�C��
?�z��,���{��C���                                    Bx�S)D  �          @s33?��ÿ�z��@  �Q33C�XR?����*�H�ff�
=C�'�                                    Bx�S7�  �          @u�?��R�����AG��P(�C��?��R�%������C���                                    Bx�SF�  �          @r�\?У��   �1G��;�HC��
?У��,(��ff��HC���                                    Bx�SU6  �          @u?�Q��Q��6ff�?C��?�Q��5��	���{C�w
                                    Bx�Sc�  �          @u�?���z��:=q�F��C�o\?���333�{��\C�˅                                    Bx�Sr�  �          @u�?�Q���
�J�H�_�RC��\?�Q��%�#33�'�C�N                                    Bx�S�(  �          @w
=?�=q��p��G��X�C�S3?�=q�1G�����{C��{                                    Bx�S��  �          @u?���33�G
=�Y=qC���?��+��{� �\C��\                                    Bx�S�t  �          @xQ�?�  ����5��=�C��f?�  �5������RC���                                    Bx�S�  �          @{�@	�����33��C�XR@	���7�������33C�l�                                    Bx�S��  �          @�Q�@�ÿ��#�
��RC���@���"�\������
=C��)                                    Bx�S�f  �          @�  @{���#�
��\C��
@{�(���(����C���                                    Bx�S�  
�          @w�@ff��
=�����C�S3@ff�   ��G��مC��                                     Bx�S�  �          @tz�@����G�� =qC���@��)���������HC���                                    Bx�S�X  �          @u�@
�H���R�(����C�˅@
�H�$z�����C�)                                    Bx�T�  �          @�G�@���\)�?\)�?�C��@�������  C�K�                                    Bx�T�  �          @}p�@
=q�����>{�F=qC�AH@
=q�z�� ����
C�R                                    Bx�T"J  �          @}p�@��+��E�M��C�
@���p��333�4�C��                                    Bx�T0�  �          @�  @=q����H���Op�C��)@=q��\)�8Q��8�RC�8R                                    Bx�T?�  �          @z�H@(�����<(��I��C�S3@(����
�#�
�(z�C�p�                                    Bx�TN<  �          @y��?�{�/\)�Q���C�=q?�{�QG�������33C�P�                                    Bx�T\�  �          @~�R?�p��#�
�*�H�(�C��?�p��J�H�����
=C��                                     Bx�Tk�  �          @��
?�{��  �U�Y��C���?�{���5��-�
C�T{                                    Bx�Tz.  �          @�{@Q쿹���S33�P33C��=@Q�����333�((�C��                                     Bx�T��  T          @|(�?xQ��a녿�\)��C���?xQ��qG��Ǯ���C�<)                                    Bx�T�z  �          @~�R?B�\�s33��\��{C���?B�\�s33?�\@�RC���                                    Bx�T�   �          @�{?��
�j=q��  ��p�C���?��
�\)�:�H� ��C�7
                                    Bx�T��  �          @�p�?���hQ��G����
C��?���}p��@  �&ffC�`                                     Bx�T�l  �          @��?���\�Ϳ�Q��ĸRC�@ ?���qG��:�H�&=qC�q�                                    Bx�T�  �          @���?�ff�Y������\)C��
?�ff�q녿z�H�_�
C���                                    Bx�T�  �          @���?�G��N�R���G�C��?�G��l(����\��\)C��=                                    Bx�T�^  �          @xQ�?L���Tz������(�C���?L���k��fff�W
=C�!H                                    Bx�T�  �          @~{>����l(���33����C�
>����{���������C��                                    Bx�U�  �          @z=q?&ff�k������{C�f?&ff�vff�#�
�ffC��\                                    Bx�UP  �          @vff?��^�R���
���
C���?��p  �
=�G�C�Q�                                    Bx�U)�  �          @z�H>��_\)���H��
=C��3>��s�
�B�\�3\)C���                                    Bx�U8�  
�          @z�H?
=q�^�R��p����HC�b�?
=q�s�
�G��8(�C��                                    Bx�UGB  �          @|(�?(���\�Ϳ����=qC�n?(���r�\�\(��H��C��3                                    Bx�UU�  �          @z�H?
=q�XQ��z���G�C��?
=q�p  �}p��j=qC�)                                    Bx�Ud�  �          @}p�?z��S�
��
��  C��3?z��n{��z���z�C�h�                                    Bx�Us4  �          @|��>���J=q�ff�p�C�c�>���e���R��(�C�9�                                    Bx�U��  �          @{�>�G��L(��Q��  C��>�G��g���G���z�C�t{                                    Bx�U��  �          @�Q�>����)���@  �A33C�� >����Tz��\)�p�C���                                    Bx�U�&  �          @z=q>�
=����I���WC�C�>�
=�>�R�\)��\C��                                    Bx�U��  �          @z=q?�\���QG��c{C���?�\�5��)���*�\C�"�                                    Bx�U�r  �          @q�>�z����?\)�Q�RC���>�z��<�����p�C�                                    Bx�U�  �          @��>���p��L(��O�C��3>���K���R�
=C���                                    Bx�Uپ  �          @�Q�>�\)�%��C�
�F��C�"�>�\)�P  ����C�|)                                    Bx�U�d  T          @u>�33�G��Mp��e�RC���>�33�0  �'
=�-(�C��{                                    Bx�U�
  T          @�  ?.{�+��5�6��C�'�?.{�S33�����C��)                                    Bx�V�  �          @~{?����8���(���C�Q�?����Y����33���C�                                    Bx�VV  �          @xQ�?���>{�{�C���?���Z�H��
=���RC�@                                     Bx�V"�  T          @vff?��?\)������C���?��XQ쿔z���ffC�k�                                    Bx�V1�  T          @x��?����>�R��
=��Q�C��f?����S33�fff�V{C��H                                    Bx�V@H  T          @vff?����@�׿�z����
C�u�?����P�׿#�
�Q�C�l�                                    Bx�VN�  T          @y��?�ff�J�H�������HC�Ф?�ff�Z=q�
=q� ��C��{                                    Bx�V]�  �          @{�?�33�HQ쿵��G�C���?�33�XQ��R��C���                                    Bx�Vl:  �          @w�?��H�K������ffC�  ?��H�Z�H�z��
ffC�C�                                    Bx�Vz�  T          @z=q?�Q��R�\����=qC�� ?�Q��^{��{���\C���                                    Bx�V��  �          @z=q?ٙ��\(��333�&=qC�  ?ٙ��`��>\)@C��                                    Bx�V�,  T          @|(�?�33�]p����R��33C��q?�33�i���\��Q�C�t{                                    Bx�V��  �          @z�H?���\(���=q��Q�C�b�?���j=q����ڏ\C�Ф                                    Bx�V�x  �          @~�R?����_\)���H���HC�s3?����o\)�
=�z�C��q                                    Bx�V�  �          @~�R?�G��`  ������\)C�
=?�G��q녿8Q��&�HC�xR                                    Bx�V��  T          @~{?�{�Vff��ff��Q�C�+�?�{�k��u�_�C�e                                    Bx�V�j  
�          @vff?.{�<(������\C�~�?.{�Z�H�У���=qC���                                    Bx�V�  T          @u?0���7��p���\C��R?0���W���(��ԣ�C��                                     Bx�V��  T          @u?#�
�9���=q�=qC�5�?#�
�X�ÿ�z����C�W
                                    Bx�W\  
�          @w
=?n{�>{��\�z�C��\?n{�Z�H���
����C��
                                    Bx�W  �          @w�?����E����H��G�C�ٚ?����\�Ϳ�Q�����C��)                                    Bx�W*�  T          @y��?���P  ��\)��p�C���?���fff����|z�C��                                    Bx�W9N  �          @w
=?(���I�����ffC���?(���c33��=q��33C�:�                                    Bx�WG�  T          @�Q�?�=q�W
=��Q����RC�S3?�=q�b�\�\���C��\                                    Bx�WV�  T          @��?��W���{����C�ٚ?��e�����z�C��                                    Bx�We@  �          @��?����W�������{C�*=?����g
=�#�
��RC�XR                                    Bx�Ws�  T          @��\?�(��U��Q�����C��?�(��l�Ϳ�\)��=qC�*=                                    Bx�W��  �          @�33?B�\�Z=q�Q����RC�N?B�\�s�
������HC��=                                    Bx�W�2  �          @���?���e����R���
C��3?���u��!G��G�C��3                                    Bx�W��  �          @��\?˅�`�׿����{C�+�?˅�n{��\��p�C��=                                    Bx�W�~  �          @��?�{�`�׿����{C�P�?�{�o\)�����
=C���                                    Bx�W�$  �          @��H?����g����\�fffC��{?����p�׾.{�
=C���                                    Bx�W��  �          @\)?�ff�`  �����C��\?�ff�i����=q�}p�C�y�                                    Bx�W�p  �          @��H?��S33��z���Q�C�
?��a녿!G��=qC�AH                                    Bx�W�  �          @�p�@�\�W
=��G����\C��)@�\�c�
����ҏ\C��                                    Bx�W��  �          @��@�U���������C�3@�fff�J=q�+33C�
                                    Bx�Xb  �          @�?���H���   ��C�j=?���`�׿�  ��\)C��                                    Bx�X  �          @�(�?�33�O\)��Q���Q�C�q�?�33�fff��z����C�K�                                    Bx�X#�  T          @��
@���\(��У���G�C���@���mp��Q��-p�C��R                                    Bx�X2T  �          @�(�?�{�p�׿��\���\C�33?�{�|(��������C��f                                    Bx�X@�  �          @�(�?�\)�qG���z��w33C�33?�\)�|(������z=qC��R                                    Bx�XO�  �          @�=q?��r�\���\�Z=qC�?��z�H����33C�b�                                    Bx�X^F  �          @���?�(��p�׿�\)�r�HC�O\?�(��z�H����b�\C��                                     Bx�Xl�  �          @���?�(��e��B�\�(z�C�ff?�(��i��=L��?(��C�"�                                    Bx�X{�  �          @���?�(��g
=����G�C�P�?�(��c33?+�AC���                                    Bx�X�8  �          @���?����`  ��33��C���?����_\)>�(�@�z�C���                                    Bx�X��  �          @��@���Z=q����h��C�h�@���XQ�>��H@�
=C���                                    Bx�X��  �          @��@
=q�Z�H����j�HC�)@
=q�Y��>��H@�\)C�5�                                    Bx�X�*  �          @���@��Z�H���
����C�N@��Vff?8Q�A'33C��R                                    Bx�X��  
�          @�33@��[�>��H@���C�0�@��O\)?�(�A�33C��                                    Bx�X�v  
�          @��\@��_\)>L��@333C��f@��W�?s33AV�HC��                                    Bx�X�  �          @�z�@Q��b�\>��
@�  C��@Q��XQ�?�=qAq�C�
                                    Bx�X��  �          @�(�@Q��a�>���@�G�C���@Q��W�?�=qAr=qC��                                    Bx�X�h  �          @��
?����j�H>\)?��RC�ff?����c33?n{APz�C��f                                    Bx�Y  �          @��?����j=q=#�
?�\C�k�?����dz�?Q�A8Q�C��
                                    Bx�Y�  �          @��?���h��=���?�=qC���?���b�\?^�RAD  C�                                    Bx�Y+Z  �          @�33?�z��fff>�33@��RC��
?�z��\(�?�{A|z�C��H                                    Bx�Y:   �          @��H?�=q�i��<��
>k�C�N?�=q�c�
?J=qA2�HC��{                                    Bx�YH�  �          @�=q?�G��j�H<�>�
=C���?�G��e�?O\)A7
=C�3                                    Bx�YWL  �          @���?�  �qG����
����C�xR?�  �l(�?G�A,(�C��{                                    Bx�Ye�  �          @�=q?���g
==L��?:�HC�Ǯ?���`��?O\)A8(�C��                                    Bx�Yt�  T          @��H?����h��=#�
?�C�w
?����c33?L��A5p�C��H                                    Bx�Y�>  
�          @�G�@ ���Z�H�
=�=qC�=q@ ���^{>��@�C�3                                    Bx�Y��  �          @���@ff�e��#�
��C�33@ff�`  ?5A{C�t{                                    Bx�Y��  �          @�ff@�
�_\)<#�
=�G�C��R@�
�Z�H?=p�A"ffC�                                    Bx�Y�0  �          @�ff?�=q�w���Q쿠  C�q?�=q�s�
?8Q�A�C�H�                                    Bx�Y��  �          @���?�
=�e�>�G�@���C�� ?�
=�Z=q?�A��C�]q                                    Bx�Y�|  �          @mp�?����N{�W
=�Q�C���?����L(�>�@�(�C��
                                    Bx�Y�"  �          @j=q?�p��C�
<#�
>.{C�xR?�p��?\)?#�
A"�RC���                                    Bx�Y��  �          @fff?���C33���R��(�C��?���C33>���@�  C��f                                    Bx�Y�n  �          @hQ�@G��?\)�\����C��@G��@  >u@uC���                                    Bx�Z  �          @j�H@Q��<�;�����
C���@Q��>�R>��@�C��)                                    Bx�Z�  �          @j�H?�p��Dz�u�s�
C�k�?�p��C�
>Ǯ@��
C�|)                                    Bx�Z$`  �          @g�?��E��33��  C�h�?��E>�z�@�z�C�b�                                    Bx�Z3  �          @h��?�Q��H�ÿ&ff�%�C�,�?�Q��Mp�    =uC��                                    Bx�ZA�  
�          @q�?�z��Mp��\)�	C�^�?�z��P  =���?�  C�/\                                    Bx�ZPR  �          @q�?�(��H�þ�p���(�C�?�(��I��>�\)@��C��)                                    Bx�Z^�  �          @qG�@��H�þ�
=��
=C�s3@��J=q>aG�@VffC�]q                                    Bx�Zm�  �          @s�
@G��C33>u@j�HC�U�@G��;�?Y��AO\)C���                                    Bx�Z|D  �          @z=q@{�@��>��
@��RC���@{�8��?k�AZ{C�K�                                    Bx�Z��  �          @z=q@#33�=p�=���?�  C�\)@#33�8Q�?333A#�
C��H                                    Bx�Z��  
�          @{�@.{�5>8Q�@+�C���@.{�/\)?@  A/�C�`                                     Bx�Z�6  �          @~�R@<(��*�H=�\)?uC��f@<(��&ff?��A
�RC�B�                                    Bx�Z��  �          @��H@G��&ff>Ǯ@�{C�
@G��{?fffAK\)C��                                    Bx�Zł  �          @��@C33�#�
?E�A.�RC��@C33�ff?�G�A�
=C�1�                                    Bx�Z�(  �          @�33@C�
�"�\?s33AW�C�.@C�
�33?�Q�A��RC��\                                    Bx�Z��  �          @���@@���=q?�(�A�{C��@@����?�A�33C�ff                                    Bx�Z�t  �          @��@#33�1G�?��A�  C�XR@#33�=q@�
A���C�U�                                    Bx�[   �          @�?�33�@��@��A��C�*=?�33�!�@,(�B!=qC�}q                                    Bx�[�  �          @�z�?��:=q@!G�B��C�?��ff@B�\B?�C���                                    Bx�[f  �          @�ff?�  �2�\@8Q�B-�C�\?�  �
�H@W�BU�C���                                    Bx�[,  �          @�p�?c�
�.{@AG�B9�C��?c�
��@_\)BdG�C��3                                    Bx�[:�  �          @�(�?\(��*=q@B�\B=�RC���?\(�� ��@`  Bh  C���                                    Bx�[IX  �          @��
?���p�@P  BQ{C�P�?�Ϳ��
@j�HB|�\C��q                                    Bx�[W�  
�          @�Q�>�
=��R@S33B]C�L�>�
=��ff@j�HB�C��                                    Bx�[f�  �          @���>�����H@L(�BRp�C��
>��ÿ�  @fffB~�C�E                                    Bx�[uJ  �          @���?(��,(�@>{B<G�C�c�?(��z�@\(�Bg�C�:�                                    Bx�[��  �          @��\?�R�+�@@  B=��C���?�R�33@]p�Bh�HC�h�                                    Bx�[��  �          @z=q>�  �(�@L��B]��C�G�>�  ���
@c�
B��
C��=                                    Bx�[�<  �          @z�H?h���'
=?��A��HC��)?h����
?�\B{C���                                    Bx�[��  �          @tz�?�(��7�������  C�E?�(��G
=�}p��p  C�.                                    Bx�[��  �          @x��?���H�ÿ�������C��\?���U��333�$��C��                                    Bx�[�.  �          @y��?�=q�E���ff���C�Z�?�=q�Tz�n{�]�C�n                                    Bx�[��  �          @|(�?�\)�E��ff��ffC��)?�\)�U��p���\��C���                                    Bx�[�z  �          @���@��C�
�У���
=C���@��Tzῂ�\�j{C���                                    Bx�[�   �          @�Q�@�\�HQ쿺�H��ffC���@�\�Vff�W
=�@��C���                                    Bx�\�  �          @\)@z��G
=�������C��3@z��Tz�B�\�0��C���                                    Bx�\l  �          @|��@�\�E�������C���@�\�R�\�E��3\)C��                                    Bx�\%  �          @~�R@33�H�ÿ�����\C��@33�U��333�"{C��=                                    Bx�\3�  �          @|��?�{�L(���33��C�&f?�{�X�ÿE��4Q�C�b�                                    Bx�\B^  �          @vff@{�>{��=q��ffC�\)@{�G���\��G�C��\                                    Bx�\Q  �          @r�\?��H�333�˅���C�w
?��H�C33���
�~ffC�XR                                    Bx�\_�  �          @o\)?����.{��33���
C�ٚ?����>�R��{����C��                                     Bx�\nP  �          @o\)?�=q�5������ffC�ff?�=q�E������C�S3                                    Bx�\|�  �          @j=q?�z��6ff���R����C��?�z��B�\�0���-C�q                                    Bx�\��  �          @h��?�(��AG������G�C���?�(��J�H��\� ��C�:�                                    Bx�\�B  �          @l��?��H�Mp��333�.�HC�
=?��H�Q녽�Q쿼(�C��                                    Bx�\��  �          @q�?�33�AG���ff��Q�C�O\?�33�P�׿xQ��mp�C�l�                                    Bx�\��  �          @j�H?�\)�>�R�����{C�H�?�\)�K��O\)�MG�C�~�                                    Bx�\�4  �          @e�@G��.{��(��޸RC��\@G��0��=��
?�(�C���                                    Bx�\��  �          @mp�?�33�<(������{C���?�33�K��xQ��r�HC�                                    Bx�\�  �          @o\)?����6ff����\)C�e?����H�ÿ��
��Q�C�B�                                    Bx�\�&  �          @o\)?��
�1G���(����C�j=?��
�E��
=���\C�'�                                    Bx�] �  �          @o\)?��R�AG���=q��G�C�#�?��R�P�׿�  �z{C�K�                                    Bx�]r  �          @q�?\�C33�У��̸RC�@ ?\�R�\��ff����C�aH                                    Bx�]  �          @p��?��
�9����=q��p�C��?��
�K����\��ffC�ٚ                                    Bx�],�  �          @l��?�=q�'
=�\)���C��?�=q�>�R��p��߅C��                                    Bx�];d  �          @p  ?����#�
�����C��=?����<�Ϳ�����C�3                                    Bx�]J
  �          @j=q?���H�ff���C�*=?��3�
��\)���
C�e                                    Bx�]X�  �          @l��@
=�/\)�8Q��4��C�Z�@
=�5��W
=�UC��                                    Bx�]gV  �          @l(�@Q��/\)�+��'\)C�z�@Q��3�
�#�
�#33C�R                                    Bx�]u�  �          @j�H@=q�,�Ϳ(��G�C�ٚ@=q�1G���G����C���                                    Bx�]��  �          @j�H@��,(��z��ffC�3@��0  ��Q쿵C��H                                    Bx�]�H  �          @k�@���-p��+��'�C��3@���2�\�.{�*=qC�N                                    Bx�]��  �          @i��@(��(�ÿ(��33C�` @(��,�;��z�C��                                    Bx�]��  �          @j�H@$z��%��   ��(�C�n@$z��(Q���
=qC�*=                                    Bx�]�:  �          @n�R@'
=�'
=��\��=qC�|)@'
=�*=q���   C�8R                                    Bx�]��  �          @o\)@(Q��&ff�z���
C���@(Q��*=q��G���(�C�P�                                    Bx�]܆  �          @o\)@+��!녿�R���C�U�@+��&ff�#�
� ��C���                                    Bx�]�,  �          @k�@)����R�!G����C�h�@)���#33�8Q��0  C�H                                    Bx�]��  �          @k�@333����G���{C��@333�Q켣�
��33C���                                    Bx�^x  �          @p��@3�
��H����ffC���@3�
�{���
��  C�S3                                    Bx�^  �          @s�
@(���p����H����C���@(���(�ÿB�\�8  C��f                                    Bx�^%�  �          @r�\@!G��#33���H��C�Z�@!G��.{�@  �6{C�h�                                    Bx�^4j  �          @p��@%��
=�����C��@%��$z�s33�k33C��\                                    Bx�^C  �          @c�
@
=��Ϳ�G�����C�� @
=��������  C��                                    Bx�^Q�  �          @g
=@z����G��ď\C�\)@z��$zῈ�����C��                                    Bx�^`\  �          @fff@
=����{�ԏ\C��q@
=��������{C�3                                    Bx�^o  �          @aG�@�
�33��
=��z�C�:�@�
��
��ff���C��H                                    Bx�^}�  �          @aG�@�R��
��\��(�C��\@�R�������C�޸                                    Bx�^�N  �          @a�@�R�ff�޸R��
=C�\)@�R��������(�C���                                    Bx�^��  �          @^�R@G���ÿ����ffC�Z�@G���������
C��)                                    Bx�^��  �          @\��@���������C�L�@��33��  ���RC���                                    Bx�^�@  �          @Z=q@ff�33��\)����C�q�@ff��׿}p����C�{                                    Bx�^��  �          @[�@��33��33���\C��@���׿��\����C�'�                                    Bx�^Ռ  �          @W
=@���(�����ffC��@��������G�C�k�                                    Bx�^�2  �          @^�R@���  ��33C�q@��
��\)����C���                                    Bx�^��  �          @c33@33��ÿ�33��
=C�K�@33�(�ÿ��H��p�C���                                    Bx�_~  �          @e�?����\)����
=C��?����1G��������
C��f                                    Bx�_$  �          @c�
?��H�(���Q��(�C���?��H�/\)���R��=qC��                                    Bx�_�  �          @c�
?�33�"�\��\)��33C�s3?�33�5���33��C��                                    Bx�_-p  �          @g
=?��%������\)C�xR?��7
=����{C�                                      Bx�_<  �          @fff?����\)��\)��C��?����1G���z����HC��                                     Bx�_J�  �          @g
=?�  ��Ϳ��R�{C�Ǯ?�  �0  ������HC�9�                                    Bx�_Yb  �          @h��?�Q��$z��Q���C��3?�Q��7
=��p����C�N                                    Bx�_h  �          @l(�?�p��&ff��p��Q�C��=?�p��9����  ��{C�b�                                    Bx�_v�  �          @p  ?�{�#33�G��(�C��?�{�7
=��ff�îC��                                     Bx�_�T  �          @o\)?��'
=��33C�G�?��;���\)��G�C��3                                    Bx�_��  �          @k�?����������C�  ?����,(���Q���z�C�=q                                    Bx�_��  �          @l(�?�ff��p��33C��?�ff�+����
����C��                                    Bx�_�F  �          @n�R?�(��p������C��R?�(��#�
��{��33C��                                    Bx�_��  �          @mp�?��!���
�Q�C���?��8Q������C��                                    Bx�_Β  �          @qG�?Ǯ�(���H��RC�K�?Ǯ�3�
��(����\C��                                     Bx�_�8  �          @p��?��
�  ��� ��C�.?��
�(Q�� ���p�C��                                    Bx�_��  �          @q�?����Q���� �\C�>�?���� ����\��HC��                                    Bx�_��  �          @q�@����{�#p�C���@���
�
=�{C�H�                                    Bx�`	*  �          @l��@�
���{�'�\C�)@�
��R�Q��\)C�\)                                    Bx�`�  �          @h��?�{�p��33�ffC�}q?�{�3�
��{��(�C���                                    Bx�`&v  �          @n{?����%�G����C�H?����<(���ff��\C��{                                    Bx�`5  �          @s33?�(��"�\�=q��C�{?�(��:=q������G�C�t{                                    Bx�`C�  �          @w
=?�  �'
=�=q�ffC��
?�  �>�R��Q�����C�ff                                    Bx�`Rh  �          @u?������$z��(��C���?���%�
=q�	�HC�T{                                    Bx�`a  �          @{�?�
=��\�/\)�1�C��q?�
=�{���\)C��                                    Bx�`o�  T          @u�?Ǯ�z��&ff�+��C���?Ǯ�.�R����C��H                                    Bx�`~Z  �          @u?�
=��H�&ff�+
=C�P�?�
=�4z��
=q�	p�C�xR                                    Bx�`�   �          @r�\?�=q����,���6�C�H?�=q��
��  C�&f                                    Bx�`��  �          @n{?�ff�ff�$z��/(�C���?�ff�/\)�	����C��3                                    Bx�`�L  �          @l��?�{���#33�/��C�o\?�{�*�H����(�C���                                    Bx�`��  �          @qG�?����
�*=q�3C��?���.{�\)�
=C�R                                    Bx�`ǘ  �          @r�\?����*�H�3\)C���?��,(��  �G�C���                                    Bx�`�>  C          @l��?�{���*=q�7�C��{?�{�%�����\C���                                    Bx�`��  �          @o\)?����
=q�.�R�;Q�C�O\?����%����p�C�{                                    Bx�`�  �          @l��?�ff�
=q�#�
�/�HC��?�ff�#33�
�H�(�C��                                    Bx�a0  �          @k�?�Q��\)�Q�� �HC�y�?�Q��'
=��(���C�xR                                    Bx�a�  �          @i��?�Q��33�   �,��C��?�Q����Q���C�g�                                    Bx�a|  �          @i��?�33��p��%�4��C�� ?�33�Q���R�\)C�W
                                    Bx�a."  T          @l(�?�녿�\)�-p��=C��H?����\���   C��                                    Bx�a<�  T          @j�H?�p������333�H�C���?�p��  �p��)z�C��q                                    Bx�aKn  �          @b�\@G��
=���H�=qC���@G������=q��{C��                                    Bx�aZ  �          @b�\?�33�����#33�7�HC�B�?�33���H����33C�
=                                    Bx�ah�  T          @i��?�������H��C�^�?��,(������  C���                                    Bx�aw`  �          @g
=?�33�(��	���
=C��f?�33� �׿�G���p�C���                                    Bx�a�  �          @e?�{�����
=�%(�C�ٚ?�{��
� ���C�o\                                    Bx�a��  �          @Y��?�z���
��=q���C��=?�z��%�����33C�U�                                    Bx�a�R  �          @W
=?�Q�����	���33C���?�Q��p���\���\C�/\                                    Bx�a��  �          @Vff?�z��ff���#G�C��?�z������Q�C��                                    Bx�a��  �          @[�?��Ϳ�p��(��5C�"�?����
=����HC���                                    Bx�a�D  �          @Y��?��Ϳ�  �*�H�P��C��?����
=q��0
=C�t{                                    Bx�a��  �          @U?k���{�.�R�_ffC��?k������=�C�1�                                    Bx�a�  �          @Vff?�z���ÿ�
=��C��f?�z����Ǯ���C�0�                                    Bx�a�6  �          @XQ�?�G������H��C��?�G��=q�˅��(�C��                                    Bx�b	�  �          @W
=�L�Ϳ���J=q��C��{�L�Ϳ˅�<(��v(�C�)                                    Bx�b�  �          @U�?fff��ff�;��v��C�L�?fff��G��+��VQ�C��H                                    Bx�b'(  �          @S33?녿����<(��{��C�E?녿��*�H�X�\C���                                    Bx�b5�  �          @P��>�׿�(��6ff�vffC�� >�׿�z��%��R\)C��                                    Bx�bDt  �          @QG�?s33���
�#�
�N��C��?s33���\)�-{C���                                    Bx�bS  �          @U?}p���(��+��U��C��
?}p��Q����4�RC�t{                                    Bx�ba�  �          @XQ�?���Ǯ�333�`(�C�
?�����R� ���@{C��q                                    Bx�bpf  �          @W�?fff���+��RG�C���?fff�  ��0{C���                                    Bx�b  �          @XQ�?k�����)���M��C���?k��33�33�+C��                                    Bx�b��  �          @U?xQ��ff�(���Q�C�!H?xQ������
�/�C�ٚ                                    Bx�b�X  �          @U�?=p��޸R�/\)�]ffC��{?=p��
=q��H�:�\C��H                                    Bx�b��  �          @S33>�
=����6ff�s�C���>�
=��p��#�
�N�HC��                                    Bx�b��  �          @W
=>��R��33�:=q�o�
C�@ >��R�ff�&ff�K
=C�%                                    Bx�b�J  �          @XQ�>���  �5�e�HC��)>��(��!G��A��C�#�                                    Bx�b��  �          @U?�\�˅�8���oz�C��?�\��\�%�K��C��                                    Bx�b�  �          @Z=q?��\�����#�
�D\)C���?��\�p���R�$�C��q                                    Bx�b�<  �          @\(�?�z����ff�,ffC���?�z��\)� ���ffC�:�                                    Bx�c�  �          @\(�?�녿�z���+�HC�O\?������   ���C���                                    Bx�c�  �          @Z=q?�ff�������,p�C�5�?�ff�33��p��Q�C��{                                    Bx�c .  �          @W�?�(�� ������(��C��?�(��ff��z��	��C��                                    Bx�c.�  �          @Vff?��R��p��  �)33C��f?��R��
��33�
�C�ff                                    Bx�c=z  �          @W�?��ÿ����
�,��C��q?����\)��(��\)C��f                                    Bx�cL   �          @U?��׿��
=�3��C��q?����G��G��C��                                     Bx�cZ�  �          @XQ�?�녿�p���R�=�C��?���ff�Q��C���                                    Bx�cil  �          @X��?�z���\����9
=C���?�z��=q��{C��                                     Bx�cx  �          @Y��?���33�{�;=qC��q?����H�
=���C��                                    Bx�c��  �          @X��?n{�z����)�C��=?n{�)�������HC���                                    Bx�c�^  �          @Vff?�����G��.{�Z�C�H�?�����
=�(��<�C���                                    Bx�c�  �          @S�
?��H�ٙ��$z��K��C���?��H�����,Q�C��                                    Bx�c��  �          @U�?�z῾�R�(Q��P  C��R?�z����ff�3z�C�W
                                    Bx�c�P  �          @W�?�33��p��*�H�Q��C���?�33��33�(��9(�C�y�                                    Bx�c��  �          @Z=q?������p��"�C�5�?�����Ϳ��(�C�W
                                    Bx�cޜ  �          @Y��?�
=��R������C��3?�
=�/\)��33�\C��q                                    Bx�c�B  �          @W�?��
��R�����C���?��
�0  ������Q�C��                                     Bx�c��  �          @W
=?���\)���G�C��f?���0�׿���C��H                                    Bx�d
�  �          @\��?�z��{��(��
�\C���?�z��0  �����C���                                    Bx�d4  �          @]p�?����&ff�����	{C�Z�?����8Q��G���{C�G�                                    Bx�d'�  �          @W
=?�G��ff�Q��6=qC���?�G��p�� ���=qC�/\                                    Bx�d6�  �          @aG�@��������C�O\@��$zῂ�\��(�C�+�                                    Bx�dE&  �          @`  ?�
=���˅��Q�C�8R?�
=�*=q����
=C�H                                    Bx�dS�  �          @^{?�=q�p���{���
C�T{?�=q�,(���Q���33C�!H                                    Bx�dbr  �          @]p�?�p���H��p���
=C�� ?�p��(Q쿈����ffC�z�                                    Bx�dq  �          @e�@��\)������p�C�1�@��,(����
���
C��                                    Bx�d�  �          @aG�@G��   ��z����C�~�@G��,(��}p���ffC�s3                                    Bx�d�d  �          @c33?�
=�=q�޸R�뙚C�T{?�
=�*=q��=q��Q�C��
                                    Bx�d�
  �          @aG�?�(��p��G���HC�g�?�(��0  �������HC�                                    Bx�d��  �          @g�?����!��Q���C���?����5��Q����C���                                    Bx�d�V  �          @h��?�  �0  �G��Q�C�0�?�  �B�\��ff��=qC�#�                                    Bx�d��  �          @j=q?�  �0  ��33���C�E?�  �AG���
=��p�C�+�                                    Bx�dע  �          @hQ�?Ǯ�1G���  ��Q�C��?Ǯ�AG����\����C���                                    Bx�d�H  �          @h��?�  �/\)��\)��(�C�P�?�  �>{��z����\C�L�                                    Bx�d��  �          @j=q?����(�ÿ�p����
C��\?����8Q쿣�
���
C�aH                                    Bx�e�  �          @c�
?�(��9����Q����\C�t{?�(��E�s33�w
=C��R                                    Bx�e:  �          @aG�?�
=�&ff��33���HC�w
?�
=�5����H��\)C�\)                                    Bx�e �  �          @e�?У��0�׿�=q��(�C�G�?У��>�R��{��33C�U�                                    Bx�e/�  
          @dz�?����&ff����ffC��{?����7�������C���                                    Bx�e>,  �          @dz�?���)����=q��Q�C�3?���:=q��\)����C��\                                    Bx�eL�  �          @h��?��5������C�J=?��Fff��ff���C�W
                                    Bx�e[x  �          @c�
?�\)�!����z�C�4{?�\)�5������=qC���                                    Bx�ej  �          @dz�?�z�����
�H��C��?�z��1G��޸R��=qC�u�                                    Bx�ex�  �          @a�?����	�����.p�C�b�?���� ���33�33C�ff                                    Bx�e�j  
Z          @i��?�p��'
=��  ��
=C��)?�p��4zῇ���z�C��                                    Bx�e�  �          @g
=@��+������z�C��\@��3�
�
=�33C��H                                    Bx�e��  
(          @fff@(��,(��n{�p��C��q@(��333����
=C�f                                    Bx�e�\  
�          @aG�?��R�(Q쿚�H��G�C���?��R�333�E��IG�C��
                                    Bx�e�  	�          @^{?�
=�(Q쿘Q����RC�(�?�
=�2�\�@  �F�HC�\)                                    Bx�eШ  T          @`  ?��H�*�H��\)���C�q?��H�4z�+��/�C�c�                                    Bx�e�N  T          @a�@��0  �Tz��[33C�+�@��6ff��p���=qC���                                    Bx�e��  "          @^�R@z��+��:�H�B{C���@z��0�׾�\)��(�C�o\                                    Bx�e��  	`          @\(�@
=�(�ÿ(��"�\C�L�@
=�-p��.{�0  C��R                                    Bx�f@  T          @^�R@ff�,(��0���5�C���@ff�1G��u�xQ�C��3                                    Bx�f�  T          @_\)@�
�.{�+��0��C��{@�
�333�W
=�^�RC�8R                                    Bx�f(�  �          @a�@
=�/\)�(���-��C���@
=�3�
�L���U�C�w
                                    Bx�f72  
�          @b�\@
=q�.�R�#�
�'�
C�%@
=q�333�B�\�A�C���                                    Bx�fE�  �          @c33@(��.{�!G��$z�C�aH@(��2�\�8Q��7
=C�
=                                    Bx�fT~  
�          @e@\)�.{�8Q��9p�C���@\)�333��=q��\)C�P�                                    Bx�fc$  
�          @j=q@�,(��L���I��C��H@�2�\��{��z�C�                                    Bx�fq�  
(          @dz�@(��#33��p���  C��q@(��%�=L��?^�RC��
                                    Bx�f�p  "          @b�\@(��"�\��\)��  C���@(��#�
>��@
=C��)                                    Bx�f�  �          @a�@����R�
=q�p�C�Z�@���!녾��33C�
=                                    Bx�f��  "          @dz�@   �\)����G�C��R@   �!녽�\)��ffC�U�                                    Bx�f�b  �          @b�\@"�\�(���{���HC�3@"�\�p�=u?��
C��                                    Bx�f�  T          @e�@�H�'����   C�^�@�H�'
=>��R@�  C�t{                                    Bx�fɮ  �          @g
=@��(�þ�{��ffC�S3@��*=q=���?�=qC�7
                                    Bx�f�T  "          @h��@#�
�$z�k��h��C�q�@#�
�%�>L��@HQ�C�n                                    Bx�f��  �          @k�@%�!G��.{�*�\C�� @%�&ff��  �{�C�p�                                    Bx�f��  
�          @l(�@$z��"�\�0���,��C��H@$z��'������  C�1�                                    Bx�gF  "          @e�@(��'�=#�
?!G�C�w
@(��%�>��@��
C��{                                    Bx�g�  T          @c33@���'
=>8Q�@7�C�E@���"�\?(�A=qC���                                    Bx�g!�  
�          @^{@�
�%�>.{@0��C���@�
� ��?
=A{C�E                                    Bx�g08  T          @`  @=q�!G�>8Q�@AG�C��H@=q���?��A�RC�B�                                    Bx�g>�  "          @dz�@{�$z�#�
�k�C��@{�"�\>�
=@�ffC�R                                    Bx�gM�  �          @e�@ ���"�\��������C�N@ ���#�
>�@C�9�                                    Bx�g\*  "          @j=q@"�\�%��
=��p�C�33@"�\�(Q�<��
>uC��                                    Bx�gj�  �          @l��@&ff�#�
�(��
=C���@&ff�'��.{�(Q�C�aH                                    Bx�gyv  "          @n�R@'��#�
�:�H�4Q�C�Ф@'��(�þ�z���ffC�XR                                    Bx�g�  T          @o\)@*=q�\)�W
=�P��C�y�@*=q�%����˅C��H                                    Bx�g��  �          @l(�@(���!G���\��C�+�@(���$zὣ�
����C���                                    Bx�g�h  
�          @e@#33� �׾������C��R@#33�!�=�Q�?�G�C���                                    Bx�g�            @dz�@!G��   �W
=�\(�C��q@!G��   >W
=@Z=qC��q                                    Bx�g´  
Z          @e@{� �׿�R��C�C�@{�%��B�\�?\)C��H                                    Bx�g�Z  
�          @a�@ff��H��  ��C�3@ff�#�
���=qC�O\                                    