CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230317000000_e20230317235959_p20230318021926_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-18T02:19:26.008Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-17T00:00:00.000Z   time_coverage_end         2023-03-17T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxn���  "          @�p��p��@l(�?���A7�
C��p��@"�\@8Q�B��C�                                   Bxn��f  �          @�p����H@QG�?��\AZffCJ=���H@�@5�A�33C��                                    Bxn��  �          @���G�@aG�?
=q@�G�C
����G�@+�@A�Q�Cp�                                    Bxn��  �          @�p�����@QG�?��Ao�C�����@G�@;�B�
Cc�                                    Bxn�X  T          @��^�R@��\?!G�@�
=Cs3�^�R@G
=@.{A�C
5�                                    Bxn�$�  �          @�ff�+�@�G�?���A`��B�{�+�@HQ�@b�\B"��C�\                                    Bxn�3�  
(          @�{�7�@��?(��@��B�\)�7�@`  @@  B(�C^�                                    Bxn�BJ  �          @�p��(�@�33?n{AffB��(�@fff@XQ�B�HB�                                     Bxn�P�  �          @��
=@���?Q�A\)B�3�
=@fff@O\)B�HB���                                    Bxn�_�  �          @�{�5@�Q�?uA"�\B�u��5@S33@N{B��C��                                    Bxn�n<  �          @��
�:�H@�G���p�����B�z�:�H@���?�Q�A|z�B�u�                                    Bxn�|�  �          @�33�#�
@�
=���
�c�
B�� �#�
@��R@.{A���B��                                    Bxn΋�  
�          @�(��z�@�p�?��
A\z�B��f�z�@l(�@uB7{B�                                      BxnΚ.  �          @���:�H@��?Tz�A
=B��
�:�H@~�R@`  B#G�BȸR                                    BxnΨ�  
�          @��H����@��>��@3�
B��=����@�33@C33B��B�ff                                    Bxnηz  
Z          @�33���@���Ǯ����B�
=���@�z�@p�A��B³3                                    Bxn��   T          @���>���@��R>L��@  B�\)>���@���@<��B	��B���                                    Bxn���  	�          @�G�>��
@��H���\�`Q�B�>��
@��?У�A���B���                                    Bxn��l  
�          @�
=?��@��\���
���B�?��@�{?�33ALQ�B���                                    Bxn��  
�          @��\��G�@�=q�z����B�����G�@}p�?�AÙ�B���                                    Bxn� �  "          @�
=���@E?�A���C����?���@QG�B�C��                                    Bxn�^  T          @���r�\@Z�H?�A��C	���r�\@   @O\)B{C(�                                    Bxn�  �          @��\�qG�@c33?�(�AS�C���qG�@
=@:�HB=qC�                                    Bxn�,�  T          @�33�w
=@e�?k�A33C	(��w
=@"�\@,(�A�\)C�f                                    Bxn�;P  �          @��\�s33@j=q?(��@�\C�s33@0  @   A��C�                                    Bxn�I�  "          @�����@Q�?=p�A�C����@Q�@�A��C�f                                    Bxn�X�  
�          @�G��xQ�@_\)?E�A=qC
��xQ�@#33@ ��A���C�R                                    Bxn�gB  �          @�=q�l(�@qG�=��
?Y��Cp��l(�@I��@z�A�ffC�=                                    Bxn�u�  "          @��H��
@�33��(���(�B��f��
@�33?Y��A{B�L�                                    Bxnτ�  
�          @�z��$z�@�ff�Ǯ��G�B����$z�@�33?��A2{B�W
                                    Bxnϓ4  "          @���(Q�@�(��c�
�  B�G��(Q�@���?ٙ�A���B��
                                    Bxnϡ�  �          @��AG�@��ͿxQ��$��B��
�AG�@�  ?�G�A�\)B�                                    Bxnϰ�  
�          @�(��aG�@{������RC޸�aG�@Vff@�
A���Cs3                                    BxnϿ&  T          @�p��_\)@�Q�?E�A�
C{�_\)@?\)@1�A��HCk�                                    Bxn���  T          @��
�8��@�(�?Q�A�\B��)�8��@Q�@AG�B
ffCff                                    Bxn��r  "          @�z��U@L��@%�A�
=CB��U?���@xQ�B<p�C�                                    Bxn��  T          @����\(�@�@P  B�C�R�\(�>�  @\)BDp�C/�                                    Bxn���  �          @�G��l��@!G�@1G�A�=qC�\�l��?+�@l(�B1��C)�                                    Bxn�d  T          @�z���Q�@0��?�33A�=qC���Q�?�(�@7
=B�\C!                                      Bxn�
  �          @�z�����@Z=q=��
?Y��C�\����@7
=?�\)A�
=Cu�                                    Bxn�%�  T          @�z���33@X��>�  @*�HCaH��33@0  @   A���C!H                                    Bxn�4V  
�          @����=q@\��=u?.{C���=q@9��?��A���C�                                    Bxn�B�  �          @��\�y��@O\)���O33CL��y��@W�?8Q�@�{C5�                                    Bxn�Q�  
Z          @���w�@i�����
�L��C�R�w�@G
=?�z�A�{C@                                     Bxn�`H  �          @�G��?\)@i��@G�A��CaH�?\)?�33@vffB=�\C�
                                    Bxn�n�  �          @�G��8Q�@vff?�{A�p�B����8Q�@�\@g
=B1�
C��                                    Bxn�}�  
          @����a�@tz�?J=qA	�C�=�a�@5�@+�A�
=CL�                                    BxnЌ:            @�G��Z�H@~�R>��H@���C�3�Z�H@HQ�@ ��A�G�C	�{                                    BxnК�  T          @����B�\@���>.{?��B����B�\@c�
@=qAمC�                                     BxnЩ�  
p          @�  �7
=@���<�>��
B���7
=@n{@Aә�B�{                                    Bxnи,  
�          @�Q��HQ�@�
=�#�
��Q�B�#��HQ�@e@p�A�
=C\                                    Bxn���  �          @����#33@�G����R�X��B�#33@���@�A�{B���                                    Bxn��x  
�          @�=q�l��@,��?p��A8��C���l��?�@{A�(�C��                                    Bxn��  T          @�Q����H?Q�@b�\B!p�C(�R���H��z�@\(�B{CC�f                                    Bxn���  T          @�
=�x��?��H@W
=B33CL��x�þ��R@p  B/�C8}q                                    Bxn�j  �          @�����=q@G�@,(�A�{C���=q?�@^�RB!�C,�                                     Bxn�  �          @������@Q�@"�\A��HC�H���?333@Z=qB�C*Q�                                    Bxn��  
�          @�{���@=q@
=qA���C
���?k�@FffB\)C({                                    Bxn�-\  
�          @����R@(��@(�A�  C�)���R?�{@P  B�C%=q                                    Bxn�<  >          @�p�����@?\)?�p�A}C�3����?��
@5�A��Cff                                    Bxn�J�  T          @�p���=q@E?��
A���C� ��=q?��H@HQ�B�\C=q                                    Bxn�YN  "          @����{�@S�
?��A�  C���{�?��H@HQ�B�Cs3                                    Bxn�g�  �          @�p��~{@R�\?�\)A���CW
�~{?��H@FffB  C�R                                    Bxn�v�  T          @�{��@N�R?��
AYCJ=��@@1�A��
Cp�                                    Bxnх@  "          @���{@W
=?8Q�@��
CG���{@\)@
=A�
=CB�                                    Bxnѓ�  T          @�p��g�@B�\@'�A��C  �g�?�p�@tz�B3�HC!E                                    BxnѢ�  
�          @��\(�@\(�@Q�A�Q�C��\(�?ٙ�@tz�B3��C��                                    Bxnѱ2  T          @���^�R@Mp�@0��A�(�C	=q�^�R?��@���B=p�Cff                                    Bxnѿ�  
p          @�
=�q�@0  @8��A�(�C��q�?W
=@y��B4��C'h�                                    Bxn��~  
          @�Q��l��@7
=@9��A��RCL��l��?n{@}p�B8z�C%ٚ                                    Bxn��$  
�          @�G�����@�H@8Q�A�
=C�����?�@n{B)�C+�                                    Bxn���  
�          @����}p�@
=q@K�B�Cff�}p�>B�\@uB0G�C10�                                    Bxn��p  T          @�G��|(�@G�@U�BC�
�|(�    @x��B2�C4�                                    Bxn�	  
�          @����r�\@  @S�
BCJ=�r�\>W
=@�  B:
=C0�f                                    Bxn��  �          @����|��@\)@G�B	�Cn�|��>���@u�B/�C/�)                                    Bxn�&b  
�          @������@�R@9��A�  C�=���>��@hQ�B#�C.aH                                    Bxn�5  �          @�\)���?���@4z�A���CxR���=�@W�BC2^�                                    Bxn�C�  T          @�p�����?�@:�HB{C\����=u@]p�B{C3)                                    Bxn�RT  
�          @�
=�xQ�@0��@)��A�RC�\�xQ�?}p�@l��B*C%��                                    Bxn�`�  �          @�\)�mp�@H��@!G�A�p�C� �mp�?��@q�B.C�                                    Bxn�o�  !          @�Q��c33@Z=q@{A�\)C.�c33?��@xQ�B2��C0�                                    Bxn�~F  
�          @����_\)@Z�H@#�
A���C���_\)?�{@|��B7=qC@                                     BxnҌ�  �          @����Q�@7�@�\A�
=Cc���Q�?��
@\(�B
=C"J=                                    Bxnқ�  
�          @�����\@(��@*=qA�\)C)���\?c�
@h��B$33C'��                                    BxnҪ8  �          @�Q��u�@\)@AG�B�HC���u�?z�@w�B4
=C+Y�                                    BxnҸ�  
Z          @���hQ�@X��?���A��C�f�hQ�?�33@Z�HBp�CJ=                                    Bxn�Ǆ  �          @�  �{�@n{?8Q�@��
C�
�{�@4z�@"�\A�33C^�                                    Bxn��*  
�          @��R���@Z�H?fffA(�C=q���@{@!�A�(�C��                                    Bxn���  �          @�\)��@B�\?c�
AffC���@	��@�A�=qC{                                    Bxn��v  "          @�\)���
@L(�?5@�ffC����
@�@\)A�{C��                                    Bxn�  �          @����Q�@R�\?�ffA0��CG���Q�@�\@%A�C�                                     Bxn��  
�          @����33@E?�  AR=qC�f��33@ ��@)��A癚C.                                    Bxn�h  T          @�\)�\)@^{?�\)Ah��C
�3�\)@�@=p�B{C@                                     Bxn�.  T          @�ff��z�@G
=?��A��HC���z�?�=q@@  BG�C(�                                    Bxn�<�  
�          @��R��G�@AG�?�(�Az�RC����G�?�@3�
A�{C�\                                    Bxn�KZ  k          @��R��{?�z�?��\AW�
C�
��{?}p�@z�A���C(��                                    Bxn�Z             @�\)��ff?�ff?8Q�@��C#h���ff?u?�z�An�HC)}q                                    Bxn�h�  �          @������\@+�?:�H@��\C�����\?�
=?��RA��HC&f                                    Bxn�wL  
�          @���Q�@1G�?xQ�A&ffCxR��Q�?��@  A�p�CQ�                                    BxnӅ�  �          @�{����@(��?޸RA���C�R����?��@5A��C"s3                                    BxnӔ�  �          @�ff��Q�@�\@ ��A�
=C)��Q�?s33@8��B \)C(#�                                    Bxnӣ>  
�          @�ff���@
=@�A�C:����?G�@4z�A�z�C*J=                                    Bxnӱ�  �          @����{?�@��A��C����{>�{@?\)B��C/��                                    Bxn���  "          @�����?�ff@33AÙ�C"#����>.{@0��A��C2                                      Bxn��0  k          @�Q�����?�@p�A��HC  ����>���@FffB	�C.��                                    Bxn���  �          @�Q���  ?��R@p�A�p�C"�H��  >#�
@*=qA�C2�                                    Bxn��|  
Z          @�  ���@1�?��A/
=C޸���?�\)@�
A��C��                                    Bxn��"  
(          @�\)����@;�?fffA�
C{����@z�@��A�p�Ck�                                    Bxn�	�  
�          @������@0��>�Q�@n{C�
���@��?ٙ�A�Q�C&f                                   Bxn�n  �          @�G����@1G�?��@���C���@�?�A��RCQ�                                   Bxn�'  �          @�33����@�?&ff@ָRC�=����?��?��HA��
C!��                                    Bxn�5�  
Z          @�33���@&ff?��A<��C�����?�@33A�Q�C ��                                    Bxn�D`  T          @��H���\@\)?��AXQ�C�����\?��R@Q�A�{C"�{                                    Bxn�S  �          @�33���?�ff>�z�@B�\C!����?�z�?�z�AAG�C$��                                    Bxn�a�  �          @��H���@�?Y��A�C�H���?�\)?��
A��\C%�                                    Bxn�pR  �          @������?�=q?:�H@�G�C#J=����?�  ?�
=An=qC)@                                     Bxn�~�  �          @�����\>���>�{@]p�C/����\>.{>��H@���C28R                                    Bxnԍ�  T          @������?
=>�33@eC-������>�{?��@�C0�                                    BxnԜD  
�          @��H���R?��>��H@��C)@ ���R?&ff?s33A(�C-5�                                    BxnԪ�  T          @�G���{?c�
>��R@L(�C*����{?�R?5@�p�C-��                                    BxnԹ�  �          @�G���p�?^�R?!G�@��HC*ٚ��p�>�G�?}p�A$z�C/aH                                    Bxn��6  �          @��
���?�\)?
=q@���C#\���?���?��\AS\)C'�R                                    Bxn���  �          @������>�׿z�H�&�\C.�f����?fff�(���
=C*aH                                    Bxn��  T          @�  ����?�>aG�@�Cٚ����?��R?�{A?�
C#�                                    Bxn��(  �          @�����H?@  ?�(�A��HC+.���H�W
=?�\)A�z�C6xR                                    Bxn��  �          @�G����=���<�>���C3  ���=�\)=�\)?@  C3G�                                    Bxn�t  �          @��\��=q����У����\C6�{��=q?
=��ff���C-�3                                    Bxn�   k          @�������aG��s33�,(�C6����>�\)�n{�)�C0                                    Bxn�.�  
�          @������;�����Ϳ��C6� ���;.{�aG���
C5ٚ                                    Bxn�=f  
�          @�����=�\)�B�\���C38R��>�G��!G���\)C.�                                    Bxn�L  o          @��������;�  �'
=C9�����Q���H���
C7�3                                    Bxn�Z�  T          @������R���Ϳ������C8:����R���
�.{��C4��                                    Bxn�iX  T          @��H��
=�Q�<�>��RC<�=��
=�:�H�\�}p�C;�{                                    Bxn�w�  "          @��H��p��E�    <#�
C<���p��+��\��  C;\                                    BxnՆ�  �          @����(��xQ�=p�����C>B���(�����\)�;�C9!H                                    BxnՕJ  
�          @�����H���Ϳ�  �P  CG����H�}p��   ���RC>��                                    Bxnգ�            @������G���\)��  CJ\)����s33�=q��C?                                      Bxnղ�  �          @����(���Q�ٙ����CI�3��(��Y���(��љ�C=�H                                    Bxn��<  9          @������׿�녿�(���=qCI�\���׿+��)����(�C;�R                                    Bxn���  =          @�=q��
=���Ϳ�p���=qCC=q��
=�L�������\)C6W
                                    Bxn�ވ  
�          @�����(����Ϳ�ff�UG�C?����(����
����{C7aH                                    Bxn��.  "          @����p��녿��R��p�CM����p���  �6ff��z�C@#�                                    Bxn���  k          @��H�����2�\�Q����RCT�3���������N�R�(�CE�=                                    Bxn�
z  �          @�33��G����Q��ˮCM��G��.{�G��C<u�                                    Bxn�   "          @�z����׿�(�����\)CGٚ���׾\�5����RC8�)                                    Bxn�'�  �          @����p����\)�b�RCMn��p���{���\)CC��                                    Bxn�6l  
�          @�z���  �Q�xQ��33CMz���  ����� �����
CE�                                    Bxn�E  �          @�z����R�\)���H�o\)CLO\���R���R�
=��z�CB�                                    Bxn�S�  T          @�{��z���\��z��=�CI���z῜(�� ����z�CAT{                                    Bxn�b^  �          @�
=��\)��
�#�
�ǮCK����\)��Q쿡G��K�
CHT{                                    Bxn�q  �          @����\)�"�\�W
=���CO
=��\)����
=�lQ�CK�                                    Bxn��  T          @��
��ff�B�\>�  @!�CT�f��ff�1녿�G��P��CR��                                    Bxn֎P  �          @��H��\)�+�?xQ�A"{CQ}q��\)�3�
��ff���RCR�3                                    Bxn֜�  "          @�������7�?0��@�33CT(�����5�J=q��CS�3                                    Bxn֫�  �          @�\)��=q�%�?�Q�Av{CQh���=q�<��<#�
=�CT޸                                    BxnֺB  
(          @�\)��\)�J=q?��Al  CXǮ��\)�\(����
�U�C[�                                    Bxn���  �          @�\)��ff�[�>aG�@�\C[@ ��ff�G
=��(��{�
CX�                                     Bxn�׎  �          @�{��(��I��=#�
>\CW�3��(��1G����R��ffCTQ�                                    Bxn��4  �          @����{�E>��
@\(�CV�{��{�7
=���H�NffCT��                                    Bxn���  T          @�Q������;�?p��A�CT�����AG��
=���CU�\                                    Bxn��  �          @�����  �G
=?B�\@��RCV����  �E�Tz��
�\CV�                                    Bxn�&  T          @�����  �*�H>\@�G�CQW
��  �!G��s33� (�CO�q                                    Bxn� �  �          @����G���>���@J�HCC���G���녾�
=���
CB��                                    Bxn�/r  "          @�{��33�c�
��z��B�\C=s3��33�#�
�.{��RC:ٚ                                    Bxn�>  T          @�����ÿ�  >�G�@�Q�CG!H���ÿ޸R��ff��=qCG�                                    Bxn�L�  T          @�����R�xQ�!G����HC>�
���R��Ϳ��\�1�C:                                    Bxn�[d  �          @��
��(��Y�������;
=C=c���(��k������k�C6��                                    Bxn�j
  �          @�����
?���(���{C-�f���
?�z῅��<��C&��                                    Bxn�x�  �          @��R�����33��\)�?\)C7� ����.{�����p�C5�)                                    BxnׇV  T          @�
=����=p�=#�
>�G�C;�=����+����
�U�C;\                                    Bxnו�  �          @���e?�{���
��(�C���e@�����33C��                                    Bxnפ�  �          @�{�1�@z�H�p�����B����1�@��
��z��G�B�                                      Bxn׳H  �          @�(��X��@�R�N{��
C�)�X��@`  ��\���
C{                                    Bxn���  9          @�33�1�@qG��"�\��B�Ǯ�1�@��׾�G����RB�#�                                    Bxn�Д            @����0��@1��Y���$  CǮ�0��@�녿����\B�W
                                    Bxn��:  T          @�����?s33���R�m�RC���@<���^{�(\)C��                                    Bxn���  �          @�p���{>�������{C)n��{@!�����V33B�
=                                    Bxn���  �          @��Ϳ�z�#�
����  CH�R��z�?\��p��v��C	s3                                    Bxn�,  �          @�33��?#�
��p�\C#�\��@1��s�
�<�C (�                                    Bxn��  �          @�p���ff@W���  ��z�B� ��ff@s33=L��?#�
B�z�                                    Bxn�(x  T          @��
�c33@'
=�:=q��C���c33@k���=q�i��C�                                    Bxn�7  "          @�  �'�@K��X�����C���'�@��Ϳ��R��{B�                                    Bxn�E�  �          @�Q��H��@S�
�C�
�p�CxR�H��@������?�
B�z�                                    Bxn�Tj  T          @�  �<��?޸R�����M��Cff�<��@c�
�>�R��C��                                    Bxn�c  "          @����!�?�����  �ZQ�Ch��!�@u�C33�\)B��                                    Bxn�q�  �          @�=q�%@�\���R�V33C�=�%@x���>�R�(�B�=q                                    Bxn؀\  T          @��\��G�>�33���
¬#�B�W
��G�@5����\�i{B�ff                                    Bxn؏  �          @�=q�!G�>�����£�3C���!G�@>�R��(��b��B�                                      Bxn؝�  �          @�Q��G�?u��ffC�{��G�@Q������C�
B�u�                                    BxnجN  �          @�
=�(��?�{�z�H�J  C���(��@^{�%���B�p�                                    Bxnغ�  �          @�Q��<��?�ff����K=qC���<��@dz��:=q� ��C�)                                    Bxn�ɚ  �          @�Q��!G�?�=q��Q��p33C�)�!G�@J�H�mp��)�HC }q                                    Bxn��@  �          @���
=q?G���
=aHC ��
=q@@����G��=��B�aH                                    Bxn���  �          @�
=�   >��H���\�x��C(��   @+������?Q�C�                                    Bxn���  �          @��R�
=>Ǯ�������C*���
=@'�����F�
C{                                    Bxn�2  T          @�\)�#33?���G��u�C(s3�#33@+���  �<�C�=                                    Bxn��  �          @��R�!�?Q����
�p\)C"{�!�@7
=�mp��0��Cp�                                    Bxn�!~  �          @�G��G�?@  ���\�C���G�@@������C�RB���                                    Bxn�0$  
�          @�  ��\>��
��p�§G�CͿ�\@2�\����j�RB��
                                    Bxn�>�  T          @�  ��Q�?�R��Q��
C�3��Q�@>�R��(��S�
B�\                                    Bxn�Mp  
�          @�  ��  ?   ��  u�C!xR��  @7����WffB�=q                                    Bxn�\  
�          @��׿�  >�=q���8RC'�q��  @,����(��e(�B�{                                    Bxn�j�  
�          @�Q쿨�ÿ!G���G�(�CMY�����?������{B�(�                                    Bxn�yb  T          @�G��˅��  ��  ��C=��˅@(������r
=B��                                    Bxnو  
(          @��\���L����\)�3C5k���@
=��p��cC&f                                    Bxnٖ�  �          @����Q�=L�����HǮC2G���Q�@\)��
=�e�HB�p�                                    Bxn٥T  "          @��
�!G�=�����|��C1Q��!G�@����z��N��Cz�                                    Bxnٳ�  "          @��ÿ��aG������CS�=��?�{��=qCc�                                    Bxn�   �          @�Q��=q�=p���(��fCI����=q?�
=�����|G�C	c�                                    Bxn��F  
�          @�
=�L(�?fff����Q�HC$8R�L(�@1G��[��Q�C�                                    Bxn���  
r          @�
=�P  ?����\)�Kz�C �f�P  @<(��P  �33C	�
                                    Bxn��  	�          @��\)?�
=�A��	ffC+��\)@Fff�������\C.                                    Bxn��8  <          @�(���p�?��
�A��	  C�
��p�@/\)�   ����C�R                                    Bxn��  l          @�����\@
�H?s33A"�HC�{���\?�  ?�=qA��C"�q                                    Bxn��  �          @�������@Q�?#�
@��
C)����?�{?��
A�
=C"J=                                    Bxn�)*  �          @�Q���ff?�p�?��\AVffC.��ff?�Q�@�A��
C&s3                                    Bxn�7�  "          @�Q���?0��?�ffA�z�C,k�����Q�?ٙ�A��C4��                                    Bxn�Fv  
Z          @�����(���?�33A@��C9!H��(��s33?O\)A33C>�                                    Bxn�U  T          @�
=��Q�?���?��
A���C'����Q�>�z�?�{A��RC0��                                    Bxn�c�  
�          @������@�
@(�A�(�C�����?xQ�@N{BffC'T{                                    Bxn�rh  �          @��R�}p�@8Q�@�A�33C  �}p�?�p�@]p�BQ�C�                                    Bxnځ  "          @�������@{@�A���C������?��
@>{BC#Q�                                    Bxnڏ�  T          @�(���{?���@'
=A�C!����{=�@>�RB�\C2k�                                    BxnڞZ  �          @�(��g�?�\)@Tz�B�HC���g��#�
@l��B6ffC4��                                    Bxnڭ   
(          @�Q��G���@���B]z�C68R�G��Q�@\)B:Q�CVL�                                    Bxnڻ�  �          @��Ϳٙ�?Tz�@�Q�B�p�C���ٙ�����@�=qB���C_W
                                    Bxn��L  "          @�z��?\(�@�(�B�aHC����\@�\)Bz=qCW�f                                    Bxn���  �          @���p�?���@�(�B��C���p���ff@��RB��\CO�                                    Bxn��  
�          @�����\?��@��HBl�C���\���@�(�B��3CA}q                                    Bxn��>  �          @����{?�@�33Bm��C�=�{��@���B�(�CA^�                                    Bxn��  T          @�(��z�?���@�G�BiffC�{�z���H@�33B�#�C?�)                                    Bxn��  �          @�z����?���@�Q�Bf�C��������@�=qB�u�C?�                                    Bxn�"0  �          @�ff�+�?��H@��BfG�Cp��+��Q�@��RBr
=CD�f                                    Bxn�0�  �          @�p��%?��@�G�BgG�C8R�%�:�H@�\)Bvp�CC�                                     Bxn�?|  	�          @�=q�)��?�=q@�Q�BY��C\)�)����{@��Bt��C;aH                                    Bxn�N"  "          @�=q�A�?��@��
BN�C�H�A녾�ff@�z�Bb=qC<}q                                    Bxn�\�  �          @�33�!G�?�Q�@��\B\�
CY��!G���=q@�
=B{�C:\                                    Bxn�kn  
�          @�p��\)@
=@�33BZ��C���\)��@��B~�C6�                                    Bxn�z  
�          @�{�'�@�
@�  BU��C�{�'���@�ffBxQ�C6��                                    Bxnۈ�  <          @�{�{?�\@��Bn�C\)�{�
=q@���B�8RCA��                                    Bxnۗ`  "          @����
=?��H@�  Bp�C	��
=��
=@��B�.C?@                                     Bxnۦ  �          @�  ��\)?���@�ffB�C�׿�\)�\(�@��
B�#�CL��                                    Bxn۴�  
�          @�\)��33?��@�z�B�C!H��33�B�\@�33B��CI��                                    Bxn��R  
�          @��R�4z�?�\)@�=qBU��Cn�4zᾙ��@�BpG�C:!H                                    Bxn���  �          @���k�?�p�@z�HB,�C���k�=��
@�z�BH(�C2��                                    Bxn���  �          @���k�?��R@s�
B)33C�f�k�>\)@�G�BEp�C1޸                                    Bxn��D  "          @�z��`  @��@w�B-z�C}q�`  >�  @�p�BNQ�C/޸                                    Bxn���  �          @�(��S33@z�@�=qB8C�f�S33=�G�@��BXG�C2)                                    Bxn��  
Z          @�p��QG�@�@��\B8Q�CG��QG�>L��@��
BZ�RC0n                                    Bxn�6  �          @�z��W�@G�@{�B0=qC\�W�>�33@���BT�C.{                                    Bxn�)�  T          @����\(�@Q�@r�\B(�CQ��\(�?�@�{BO��C+xR                                    Bxn�8�  
          @���fff@�@fffB {C)�fff?��@�Q�BE�C+E                                    Bxn�G(  
�          @���j=q@p�@c�
B=qC޸�j=q>�@��BA�C,                                    Bxn�U�  
t          @��
�u?�z�@g�B C���u>.{@��HB;(�C1u�                                    Bxn�dt  
@          @�{����?�=q@fffB��C������=�G�@���B4Q�C2k�                                    Bxn�s  n          @����u?��@o\)B%z�C�u<��
@�z�B<�C3�                                    Bxn܁�  �          @��\�~{?��@j�HB$�
C!W
�~{��Q�@x��B1=qC9)                                    Bxnܐf  �          @������?�ff@W�Bz�C�����׼�@n{B$33C4p�                                    Bxnܟ  
�          @����?��@dz�BG�C%\)����@l(�B%  C;B�                                    Bxnܭ�  �          @�G��\)?s33@n{B(��C&���\)�8Q�@qG�B+��C>G�                                    BxnܼX  
�          @��\�~{?xQ�@r�\B+Q�C&@ �~{�:�H@vffB.\)C>ff                                    Bxn���  �          @�z��|��?�G�@x��B.C%���|�Ϳ=p�@|��B2�C>�{                                    Bxn�٤  
�          @�ff�n�R?��
@�33B8��C ���n�R��@�Q�BB  C<�                                     Bxn��J  "          @��R�Vff?�
=@��BIC�
�Vff�\)@�33BVG�C=�{                                    Bxn���  �          @����I��?�ff@�=qBQC޸�I����@���Ba33C=u�                                    Bxn��  �          @�  �C�
?Ǯ@�33BU
=C��C�
��@��\Be
=C=��                                    Bxn�<  T          @���<��?�
=@�{B\
=C@ �<�Ϳ.{@�33Bh  C@�                                    Bxn�"�  �          @�\)�2�\@\)@�BL33C=q�2�\>.{@��RBrQ�C0p�                                    Bxn�1�  T          @�\)� ��@'�@�BJz�C�\� ��?�@��
B}ffC(@                                     Bxn�@.  �          @�\)�%@
=q@��
BWQ�C@ �%<�@�33B|
=C3h�                                    Bxn�N�  �          @���5?�\)@�G�Bb��CE�5�B�\@�p�Bl��CB�q                                    Bxn�]z  
�          @���O\)?��@��BU�RC!z��O\)�p��@�z�BW�CD!H                                    Bxn�l   T          @�\)�QG�?��R@��RBL��Cz��QG��   @�p�BZ�C<�                                    Bxn�z�  �          @��<(�?���@�=qBW��C��<(���@��Bi
=C=B�                                    Bxn݉l  
�          @�{�l(���z�@��BE\)C8� �l(��33@s33B'�HCQ�                                    Bxnݘ  "          @��R�x��    @�B<\)C3�R�x�ÿ�p�@s33B'33CL�                                    Bxnݦ�  T          @����=q��@o\)B#C4\)��=q��=q@Y��B�CH\                                    Bxnݵ^  	�          @��R��p�>�
=@w
=B*�\C.0���p����H@l(�B!p�CD0�                                    Bxn��  
�          @�p���(��\@QG�B�CH:���(��/\)@ffA��CU�=                                    Bxn�Ҫ  n          @�
=��\)��@W
=BQ�CK����\)�Dz�@33A�Q�CW�                                    Bxn��P  
�          @�����H�E�@vffB*�C>�R���H��\@L(�B��CQ:�                                    Bxn���  
�          @�Q��~�R>��@�(�B7��C0:��~�R���H@w�B)�CH
                                    Bxn���  
�          @�{�g
=?��@���BE�C*��g
=����@�z�B=��CFh�                                    Bxn�B  "          @�{�p  ?��@r�\B(ffCk��p  >#�
@�\)BA�C1��                                    Bxn��  
�          @��R�p  ?��@w
=B*Q�CY��p  >\)@�G�BC=qC1޸                                    Bxn�*�  �          @��R����?�
=@w�B*��C#�����׿   @�Q�B2�HC;�                                    Bxn�94  
Z          @�
=�|��?�\)@x��B+��C ���|�;���@��B8Q�C8�\                                    Bxn�G�  l          @�
=��?���@\(�B��C%�3����p�@fffB
=C8                                    Bxn�V�  <          @�\)��  ?���@N�RB	��C"����  �L��@aG�B
=C4�f                                    Bxn�e&  �          @�{��=q?���@\��B\)C"�{��=q�#�
@l��B"p�C6(�                                    Bxn�s�  "          @��R����?�Q�@`  BG�C!ff���׽���@q�B&
=C5aH                                    Bxnނr  �          @��R�tz�?��@��HB7�C$��tz�(��@�B<��C=�{                                    Bxnޑ  �          @�ff�n{?��@��B;�
C"���n{�#�
@���BB=qC=�3                                    Bxnޟ�  �          @�{�w�?��
@|��B0�\C!�3�w���
=@�(�B:��C:@                                     Bxnޮd  �          @��i��?��@�(�B;  C 5��i����@��BE�HC;#�                                    Bxn޽
  
Z          @��R�k�?���@��B5��C��k��B�\@��HBF��C7                                    Bxn�˰  �          @��R�n�R?�
=@��B9��C"n�n�R��@��BAQ�C<�f                                    Bxn��V  �          @�
=����?�\@g
=B�Cn����>.{@�Q�B2�
C1��                                    Bxn���  �          @�ff�x��@   @j�HB {C� �x��>�p�@��B;p�C.�)                                    Bxn���  T          @��R�fff?���@|(�B/�RC���fff>W
=@�z�BJ�\C0�{                                    Bxn�H  
�          @�
=�u�?��H@n{B#��C�3�u�>��R@�{B>{C/c�                                    Bxn��  �          @�\)�U?���@���BI��C�H�U��(�@��
BWffC;E                                    Bxn�#�  �          @�Q��G�?�{@��BZffC k��G��Tz�@��RB^p�CB�3                                    Bxn�2:  
�          @�G��J�H?�
=@�{BX�
C���J�H�G�@���B^z�CAٚ                                    Bxn�@�  
�          @����B�\?\(�@��Bc��C$+��B�\����@���B_��CHxR                                    Bxn�O�  �          @�33�QG�?�p�@�33BO�HC�)�QG���ff@��B]��C;�H                                    Bxn�^,  	�          @��\�E�?�=q@�BV
=C��E��Ǯ@�p�Bg(�C;@                                     Bxn�l�  	�          @��\�G
=?��@�\)BY{C�f�G
=�
=@���Bd��C>�3                                    Bxn�{x  �          @�G��G
=?��@�
=BK�C���G
=��@�33Bep�C4�\                                    Bxnߊ  �          @���>�R@(Q�@�33B7z�C
���>�R?O\)@���Bd��C$�{                                    Bxnߘ�  	�          @�  �5�?�33@��RB\Cٚ�5�����@�\)Bp��C:�R                                    Bxnߧj  
�          @����+�?�z�@�z�Bi�CB��+���R@���Bu�HCA
=                                    Bxn߶  
�          @�  �
=q?.{@��
B�33C"xR�
=q��\)@�\)B{�CTaH                                    Bxn�Ķ  T          @�Q���?8Q�@��
B~�C#�{�����@��Bs��CP�
                                    Bxn��\  �          @���%?E�@�G�Bw33C#���%��G�@�{Bo
=CM�H                                    Bxn��  
�          @�Q��E�?��@���BY�C\)�E���R@���Bb��C?c�                                    Bxn��  �          @�Q��mp�?�33@}p�B-��Cٚ�mp�>aG�@�(�BF�
C0��                                    Bxn��N  �          @�������@0  @*�HA�Q�C8R����?�  @a�B�C ��                                    Bxn��  T          @����Q�@P��?�A�\)C!H��Q�@
=@333A�Q�CY�                                    Bxn��  :          @�=q���\@Q�?�(�Aip�C^����\@p�@'�A��C�q                                    Bxn�+@  �          @�����@>�R?��HA��Cc����@@.�RA��HC�
                                    Bxn�9�  
�          @�z��u@�
@Z=qB33C���u?Q�@�G�B7\)C'��                                    Bxn�H�  �          @�{���@\��>�
=@�G�C�q���@AG�?��HA�  C�
                                    Bxn�W2  
�          @����z�@l(�?���AD  C
Q���z�@<(�@!�A�p�C�
                                    Bxn�e�  "          @�Q��}p�@��?}p�A33CG��}p�@W�@{A˅C��                                    Bxn�t~  
�          @�\)��33@i��?333@�G�C
=��33@Fff@�\A�33C�                                    Bxn��$  ;          @�  ��z�@@  ���R�E�Cu���z�@8��?\(�A	��Cff                                    Bxn���  
�          @�Q�����@n�R�u���C������@^{?�{AZ�HC�R                                    Bxn�p  m          @�Q���(�@n{>�=q@*�HC�f��(�@U?�z�A�ffC�                                    Bxn�  
�          @�\)�w�@��R�.{��
=C���w�@|��?�Q�Aip�Cff                                    Bxnོ  
�          @�  �b�\@������C :��b�\@�33?�(�AD��C)                                    Bxn��b  
A          @�\)�hQ�@��ÿ޸R��z�C  �hQ�@�zὸQ�c�
C�{                                    Bxn��  �          @�p��[�@x�������ffCk��[�@�ff�!G����HB�8R                                    Bxn��  
s          @�����@dz��ff�~�RC
=���@x�ý�Q�p��C�{                                    Bxn��T  	          @��n�R@\)�\�x  C{�n�R@���=���?}p�C+�                                    Bxn��  T          @��y��@w
=���R�s33CJ=�y��@�(�=�\)?0��CL�                                    Bxn��  
�          @��R�g
=@z�?�A�=qCE�g
=?�\)@�A�33C�
                                    Bxn�$F  �          @��
�Vff>�@�G�B\  C1�)�Vff��
=@��BH��CN�{                                    Bxn�2�  
Z          @��
�z=q>�p�@��HB?ffC.�\�z=q���\@��B5ffCF�                                    Bxn�A�  
�          @�(��n�R�L��@�G�BJp�C4�\�n�R�޸R@�{B6(�CM�                                    Bxn�P8  	          @�(��K��:�H@��
B`�
C@���K��\)@�  B9��CZ!H                                    Bxn�^�  
Z          @����qG�?#�
@�\)BF
=C*Q��qG�����@�z�BA33CC�\                                    Bxn�m�  T          @��
�~{?���@��HB3p�C$��~{��@�ffB9\)C;�                                    Bxn�|*  
�          @����
=?��@e�BQ�C%�q��
=��=q@p  B�C7xR                                    Bxn��  
�          @�  ���?�G�@0  A�p�C�����?(�@Mp�B�C,Q�                                    Bxn�v  �          @�  ���\?�33?��RAqC%L����\?@  ?�33A�p�C+�q                                    Bxn�  T          @�  �Z�H@@q�B)�RC�
�Z�H?G�@�(�BM\)C'!H                                    Bxn��  
�          @�G��c33@E�@1G�A��C�c33?���@mp�B+G�Cz�                                    Bxn��h  T          @��H�#�
@��H�0����{B�녽#�
@��?�p�A]��B��                                    Bxn��  T          @��ͿTz�@���?0��A�B��)�Tz�@o\)@{A�=qB���                                    Bxn��  T          @��U�@�H@n{B(�\C��U�?c�
@�33BNp�C%{                                    Bxn��Z  "          @�\)�o\)@p�@[�B�
C���o\)?��@�33B:{C$xR                                    Bxn�    �          @�p��g�@G
=@?\)B (�CO\�g�?�@z�HB0G�Cp�                                    Bxn��  T          @�ff�hQ�@r�\@��A�  C��hQ�@0  @XQ�BQ�C޸                                    Bxn�L  �          @�Q��E�?333@���Bb=qC'&f�E���=q@��RB](�CGaH                                    Bxn�+�  T          @����HQ�?�33@��
BU��C��HQ쾽p�@��Bc(�C:��                                    Bxn�:�  
�          @��s�
?�  @z=qB.�Cs3�s�
�#�
@�{B>�HC40�                                    Bxn�I>  T          @��
�i��@u?�ffA���C���i��@;�@Dz�B(�C=q                                    Bxn�W�  	�          @�=q�_\)@�(�?�Q�AE�C8R�_\)@Z�H@%A߮C��                                    Bxn�f�  �          @��
�E@&ff@vffB.\)C���E?��@���BX�
C!Y�                                    Bxn�u0  �          @��{@U@g
=B${B��)�{?���@�=qB`��C�{                                    Bxn��  
�          @�
=�޸R@\)@l(�B!�B�G��޸R@��@��
Bj�B�\                                    Bxn�|  �          @�{�z�@�(�@FffB��B�3�z�@0  @��
BJ{C33                                    Bxn�"  T          @�{�;�@��H@(Q�Aܣ�B�B��;�@8Q�@z=qB.z�C��                                    Bxn��  	          @��\�_\)@E�@   A�=qC
�
�_\)?�p�@\(�B"Q�Cff                                    Bxn�n  
s          @��H�c33@<(�@5�A�z�Cn�c33?޸R@l(�B,�C��                                    Bxn��  
�          @����1G�@w
=@.�RA�G�B�G��1G�@(��@z�HB7
=CaH                                    Bxn�ۺ  �          @�33�H�þ�33@�33BW�
C:O\�H�ÿ�\)@|(�B<��CR��                                    Bxn��`  �          @��g
=�u@��BIp�C4�H�g
=����@��B7�CK�
                                    Bxn��  
�          @��R�c�
>\)@�BL�RC1�=�c�
��Q�@�{B>
=CI��                                    Bxn��  �          @���u�?W
=@��HB8�C'���u��(��@�(�B:�HC=��                                    Bxn�R  �          @�  ��(�?�=q@s33B&�RC%h���(�����@|(�B.{C8��                                    Bxn�$�  "          @�Q���{@�@.�RA�Cٚ��{?�G�@X��B{C$�                                    Bxn�3�  
�          @����(�@��@:=qA���CL���(�?��@`  Bp�C&�H                                    Bxn�BD  
Z          @�����33?�p�@W�B
=CY���33>�G�@p��B"�
C.E                                    Bxn�P�  
Z          @�����ff@�@AG�A���C����ff?fff@c�
BffC(�H                                    Bxn�_�  T          @�G�����?�{@C�
B 
=C�����?+�@aG�B�C+�=                                    Bxn�n6  �          @�G���(�@"�\@0��A��C�H��(�?�@^{B
=C"�                                    Bxn�|�  �          @�����G�@)��@5�A�(�CW
��G�?�  @dz�B��C �                                     Bxn㋂  
�          @������H@(Q�@1G�A�
=C�q���H?�G�@`��B��C �{                                    Bxn�(  T          @�����G�@*�H@
=A��HC����G�?�Q�@H��B�C��                                    Bxn��  �          @�Q�����@��@z�A��\C� ����?��
@1G�A�(�C"O\                                    Bxn�t  �          @��R��z�@��?�\)A��CaH��z�?޸R@
=A�  C ff                                    Bxn��  T          @�Q���=q@�?���A�p�C33��=q?��
@=qA�C%޸                                    Bxn���  
�          @�����{@Q�?���AX��C����{?��@ ��A��\C#��                                    Bxn��f  	          @�Q����@�\>�Q�@hQ�C.���?��?��A'33C!�=                                    Bxn��  ;          @�����  ?޸R?\)@��C"p���  ?�Q�?�\)A3
=C%L�                                    Bxn� �  �          @�����\)?��H?8Q�@�p�C"�R��\)?�{?�G�AI��C&)                                    Bxn�X  �          @�����  ?�=q?xQ�AffC$���  ?�33?���Ag�
C(33                                    Bxn��  �          @��
����?��H?xQ�AQ�C"� ����?��
?�  Alz�C&�                                    Bxn�,�  T          @��H���?�
==�G�?�=qC �
���?��
?B�\@�Q�C"                                    Bxn�;J  �          @�33��Q�?���?z�@��C!s3��Q�?��?�A8��C$Y�                                    Bxn�I�  
�          @�z���p�@.�R?5@�z�C33��p�@�?˅AzffC�=                                    Bxn�X�  �          @�ff��p�@9��>�@�(�C�3��p�@$z�?�
=A_
=C�
                                    Bxn�g<  T          @�ff��@
=��(���C}q��@�>Ǯ@qG�Cn                                    Bxn�u�  
Z          @�{����@\)�u�Q�C�\����@*�H�u��CE                                    Bxn䄈  �          @������@\)��(���Q�C�����@,(��^�R�  Cu�                                    Bxn�.  
�          @�33��?�����G�C�)��@�H����((�C�                                    Bxn��  ;          @������@�Ϳ�\)�b�RCE���@1G�������HCY�                                    Bxn�z  
�          @�z���  @mp��=p���\)Cz���  @p  ?��@�=qC:�                                    Bxn�   
�          @�����\@n�R��ff�&�HC=q���\@w�>�  @ ��C
5�                                    Bxn���  T          @�����@:=q�����
=C���@QG���\���C�R                                    Bxn��l  "          @�Q���z�?�33�33���C�{��z�@#33����x��Cu�                                    Bxn��  T          @�\)��G�@�ÿ��H���\C���G�@9����ff�*=qC��                                    Bxn���  
�          @�\)��p�@6ff��R���
C33��p�@Z�H��z��;33C5�                                    Bxn�^  	�          @�G����@L�Ϳ����=qC�����@i���5��33Cff                                    Bxn�  �          @�Q���33@��,����Q�CL���33@8�ÿ�����(�C޸                                    Bxn�%�  
�          @�����H@\)�=p���  C�f���H@E��
���HCz�                                    Bxn�4P  
�          @���=q@-p����R���\C���=q@Mp��}p��&=qC^�                                    Bxn�B�  �          @�33���@N{?8Q�@��C�����@333?޸RA�=qC=q                                    Bxn�Q�  T          @���U@�Q��	�����RC�\�U@���5��B�L�                                    Bxn�`B  �          @��R�<��@�Q�������B�aH�<��@����Y�����B��                                    Bxn�n�  ;          @�{�l(�@u�����\)C���l(�@�p����
�UC�                                    Bxn�}�  m          @��R��{@g
=��{�]G�CG���{@w
=��G���{C	ff                                    Bxn�4  �          @�ff�|(�@dz��{��  C	�\�|(�@~�R�
=����C�3                                    Bxn��  �          @�ff�G�@�zῢ�\�U�B��G�@��>aG�@�\B���                                    Bxn婀  �          @����   @�z��  ��33B�33�   @�{��\)�(��B�u�                                    Bxn�&  �          @���>�R@���33��
=B�(��>�R@���@  ��p�B�z�                                    Bxn���  �          @��\�E�@�������B�=q�E�@�33�h���z�B�Ǯ                                    Bxn��r  
�          @�33�E@y���?\)���
C W
�E@�������z�HB�\                                    Bxn��  �          @�=q�:�H@i���]p���HC ���:�H@�=q�����B��                                    Bxn��  "          @���Z=q@_\)�O\)�p�CT{�Z=q@�33������p�C )                                    Bxn�d  
Z          @��H�a�@L(��XQ����C	��a�@�33����C��                                    Bxn�
  �          @�=q�vff@U�333��Q�C
�q�vff@�=q��=q�|z�Ch�                                    Bxn��  �          @������@%��G
=��HC�����@\(������
CY�                                    Bxn�-V  "          @���h��@\(��&ff�ۮC���h��@��H��\)�`z�C�3                                    Bxn�;�  
�          @����Vff@s�
�,(����
C^��Vff@�
=����X  B�Ǯ                                    Bxn�J�  �          @�G��<(�@{��B�\��{B��\�<(�@�ff�У���33B��                                    Bxn�YH  T          @����I��@�G���R���HB��f�I��@��
�����-B���                                    Bxn�g�  
�          @�  ���@<(��B�\��z�C�H���@5?@  @��HC�                                    Bxn�v�  �          @�������@{�^�R�
�RCY�����@'��#�
��G�C                                    Bxn�:  T          @��\����@N�R����QG�C8R����@^�R�k��  C0�                                    Bxn��  �          @������
@g��\�o�
CT{���
@z�H�����;�C
�                                    Bxn梆  
�          @�{��@P���(Q���33C
��@z�H��(��f�\C�)                                    Bxn�,  "          @�p���Q�@`  �   ��G�C
���Q�@�33���\�HQ�CY�                                    Bxn��  
Z          @�����Q�@
�H�U��z�CJ=��Q�@Fff��R���HCs3                                    Bxn��x  T          @������\?��
��z��.{CxR���\@0  �]p��=qC\                                    Bxn��  "          @�����?�������)�RC=q���@5��U����C�                                    Bxn���  �          @��
�q�?�=q���R�7=qCY��q�@3�
�`���CY�                                    Bxn��j  T          @���R�\?��\���V��C"ٚ�R�\@�H��(��5G�C��                                    Bxn�	  
A          @�=q�`��?������R�I{C!��`��@!��w��'(�C.                                    Bxn��  l          @����Vff?��\��Q��N�C0��Vff@'
=�x���*
=C
                                    Bxn�&\  �          @�ff�]p�?�(����\�F��C ���]p�@   �o\)�$��C+�                                    Bxn�5  
�          @�{�W�?����=q�Ep�Cn�W�@333�hQ���\CG�                                    Bxn�C�  
t          @�{�C�
?��R��=q�UQ�C\�C�
@4z��xQ��,�C	G�                                    Bxn�RN  	�          @��B�\?�����(��\�C ���B�\@(����\�9�C=q                                    Bxn�`�  T          @����2�\?s33����i�HC!=q�2�\@Q������E��C��                                    Bxn�o�  �          @���Fff?��
��p��E��C#��Fff@>{�Z�H�=qC0�                                    Bxn�~@  	�          @�G��B�\@\)�y���3Q�C�3�B�\@c�
�<��� ��Cu�                                    Bxn��  �          @��H�N�R@G���=q�;\)C���N�R@J�H�P  ���C�{                                    Bxn盌  �          @�=q�W�@G��|���4z�C
�W�@HQ��H���	\)C	&f                                    Bxn�2  �          @���W�?�\)�\)�7��C���W�@@  �N�R�p�C
Y�                                    Bxn��  
�          @����K�?�Q����\�>ffC���K�@E�S33��\C޸                                    Bxn��~  �          @����-p�@p�����A33C	�\�-p�@e�J�H���B�#�                                    Bxn��$  �          @����(��@�H��ff�F�C	n�(��@e��QG��(�B��q                                    Bxn���  �          @�  ��@*�H��{�G  C#���@s�
�L(���HB��                                    Bxn��p  T          @�G��J=q@Q��u�0��C��J=q@[��<(��  C��                                    Bxn�  �          @����R�\@
�H�x���2��C� �R�\@P  �C�
���CT{                                    Bxn��  
�          @���p�@B�\�tz��133C  �p�@����.{��G�B�                                    Bxn�b  �          @������@
=�<(��{C
=���@9���
�H���C{                                    Bxn�.  T          @�G�����?�p��33����C �����@G���Q��r�RC�)                                    Bxn�<�  n          @�
=�w�@��N�R�=qC���w�@=p��p���=qC��                                    Bxn�KT            @�(��P  @G��q��2�RC&f�P  @C�
�@����C�q                                    Bxn�Y�  
�          @��\�X��@	���^{�#z�C���X��@E�*�H���C	�f                                    Bxn�h�  T          @�  �{�@p��ff����C��{�@C33��G����HC.                                    Bxn�wF  
�          @�����33@��=u?:�HCxR��33@�?8Q�A ��C��                                    Bxn��  �          @�����\)?z�k��p�C-�R��\)?�R�#�
���HC-E                                    Bxn蔒  
�          @��H��G�>L�Ϳ����(�C1�H��G�>�33�����C08R                                    Bxn�8  	�          @�z���=q>��
�#�
�ۅC0�
��=q>��H������C.��                                    Bxn��  
Z          @�z���Q�aG��:�H��33C6n��Q�    �B�\�\)C3�q                                    Bxn���  T          @��H���\?s33�����C)h����\?���k�� ��C(+�                                    Bxn��*  �          @�ff��(�?ٙ������{�C ��(�@�
�c�
�Q�C&f                                    Bxn���  	�          @�{��?ٙ��!G���C����@���z���ffC�H                                    Bxn��v  �          @�(����?�\)�7
=�G�C"n���@�������C�\                                    Bxn��  �          @�z���Q�?�ff�(�����
C���Q�@   �   ��=qC�f                                    Bxn�	�  
B          @����
@(���z���Q�Cff���
@   �333���RC0�                                    Bxn�h  �          @�������@G�?���A���C\����?�33@�A���C#�f                                    Bxn�'  �          @������?�z�@��A�Q�C33����?���@5A��C&                                      Bxn�5�  
Z          @�\)���?�p�@.�RA��C�H���?\(�@G�B  C)                                    Bxn�DZ  
�          @�������?�Q�@&ffA�{C������?\(�@>�RB\)C)@                                     Bxn�S   T          @�G���=q?��R@0��A�C$���=q>Ǯ@@  B�C/�                                    Bxn�a�  �          @�G����H>�z�@Z�HB�C0(����H�5@W
=B��C=T{                                    Bxn�pL  
�          @�33���?
=q@n�RB(
=C,������\)@n�RB'�
C;��                                    Bxn�~�  T          @�����G�?�=q@R�\BffC%����G�=�Q�@]p�B�C2��                                    Bxn鍘  �          @�  ���
?��@E�B�C^����
?k�@_\)BQ�C'aH                                    Bxn�>  T          @���S33?���@o\)B1(�CaH�S33?J=q@�z�BJ�RC&u�                                    Bxn��  �          @����HQ�?��
@���BT��C!�{�HQ쾏\)@�Q�B\��C9�                                    Bxn鹊  �          @����=p�?�
=@�{BGffC�{�=p�?&ff@�=qBa��C'�{                                    Bxn��0  
�          @�=q��z�?�
=@�=qB�G�C33��zὣ�
@���B�C6                                    Bxn���  T          @����
=?ٙ�@���B�Q�C����
=>B�\@��\B�C-��                                    Bxn��|  T          @���	��?�G�@�{Bw�
C���	��=#�
@��B��RC2��                                    Bxn��"  "          @��H�w�?��R@FffB�HC���w�?��\@b�\B%�HC%5�                                    Bxn��  
�          @��H�8��?�  @L��B6
=C�f�8��>�z�@[�BF�HC.\)                                    Bxn�n  �          @�z��!�    @���B}  C3��!녿�@��Bk��CQY�                                    Bxn�   T          @���� �׽���@�G�B~(�C6=q� �׿\@��Bjp�CSG�                                    Bxn�.�  "          @�{��þ�=q@���B��C:T{��ÿ��H@��BkffCW��                                    Bxn�=`  T          @����,(�?�\@���Bp�RC):��,(��\(�@�\)Bl��CE�R                                    Bxn�L  �          @���_\)?�
=@{�B2ffC{�_\)?=p�@�=qBI�C(                                    Bxn�Z�  T          @��\�`��?��R@uB.ffC� �`��?Q�@�  BF��C&��                                    Bxn�iR  T          @�
=�^�R?��R@i��B)G�CL��^�R?aG�@��BB
=C%�{                                    Bxn�w�  �          @�z��_\)?�@|(�B3�HC5��_\)?(��@���BIC)Q�                                    Bxnꆞ  �          @�{�G�?�@�(�BY(�Ch��G��.{@���BcG�C75�                                    Bxn�D  T          @���5�?�33@��Be{C�f�5��k�@�(�BoQ�C8�
                                    Bxn��  T          @��
�{?���@��RBc�\C���{>�(�@���B}��C*�                                    Bxn겐  
�          @�(��?�p�@�Bwp�CG���L��@��\B�u�C8�)                                    Bxn��6  �          @�(��'
=?���@��Bg(�C��'
==�\)@��RBx�HC2c�                                    Bxn���  T          @����{?���@�=qBj33C{�{>L��@�=qBC/n                                    Bxn�ނ  
Z          @�G���
?�ff@��Bqp�C
�=��
>���@��
B�z�C)�                                    Bxn��(  
�          @�  �p�?˅@�Q�Bp�HC0��p�>W
=@���B�G�C.��                                    Bxn���  
�          @����R?�{@���Bk
=C
��R>�z�@���B��)C,z�                                    Bxn�
t  "          @������?�
=@��RBh�\C�����>��
@��B�\)C,�                                     Bxn�  �          @�33��?�
=@��
Bqp�C�3��<�@�=qB��)C3O\                                    Bxn�'�  
�          @���&ff?�=q@���Bj33C��&ff�#�
@��RByG�C4�)                                    Bxn�6f  "          @����2�\?�G�@��Ba�
C���2�\�u@���Bo{C533                                    Bxn�E  
�          @���J=q?p��@�
=BV�\C#s3�J=q����@��B\��C9�                                    Bxn�S�  n          @�G��"�\?aG�@��HBt  C �
�"�\��@���Bx�HC>�                                    Bxn�bX  :          @����	��?�R@��B�L�C$�	���J=q@���B�
=CHG�                                    Bxn�p�  
Z          @�G��G�?c�
@��RB=qC���G��   @���B�33C@u�                                    Bxn��  T          @����{?\(�@��HB�#�CO\��{��@�(�B�p�CD�3                                    Bxn�J  T          @�
=� ��?&ff@���B�ffC"�� �׿@  @�Q�B��3CHY�                                    Bxn��  
�          @�{��G�?xQ�@N�RBp�C&s3��G�=u@W�Bp�C333                                    Bxn뫖  
�          @��R�xQ�?��H@W�B(�Ck��xQ�>��H@h��B+�C,�
                                    Bxn�<            @�Q��j=q?+�@�G�B=�C)�{�j=q��ff@�=qB?p�C;�                                    Bxn���  �          @����q�?��\@{�B4=qC$�
�q녽��
@��B<  C50�                                    Bxn�׈  n          @����vff?!G�@xQ�B3\)C*���vff��G�@z=qB4�C:��                                    Bxn��.  �          @�Q���{=�@e�B"  C2W
��{�Y��@^�RB�C?}q                                    Bxn���  T          @����(�>k�@j�HB&�\C0�\��(��E�@fffB"p�C>�{                                    Bxn�z  T          @��\���
>k�@n�RB(�C0� ���
�G�@j=qB$\)C>�                                     Bxn�   
�          @��\��Q�?&ff@a�BG�C+W
��Q쾨��@e�B��C8aH                                    Bxn� �  
�          @�(��~{?���@o\)B'(�C!5��~{>�z�@~{B3��C/�=                                    Bxn�/l  �          @�=q�`�׽�Q�@���BC33C5��`�׿�@uB8=qCFff                                    Bxn�>  "          @�(��z�H>�ff@\)B5Q�C-u��z�H�#�
@~{B3�
C==q                                    Bxn�L�  T          @���|��?0��@w
=B/��C*
=�|�;�p�@z=qB2Q�C9Q�                                    Bxn�[^  
�          @����p�<��
@Dz�B(�C3����p��L��@=p�A��
C=Ǯ                                    Bxn�j  �          @��H�G
=?�Q�@���BM�C(��G
=>u@���B]�HC/�f                                    Bxn�x�  
�          @��H��=q>�@_\)B{C-����=q���@_\)B33C:(�                                    Bxn�P  T          @�33�����p�>�\)@G�CPc�������;\��  CP@                                     Bxn��  "          @�p����@L��@uB<�HB�p����@�
@��Br  B��=                                    Bxn줜  T          @�(����
@?\)@w�BD  B�
=���
?�{@���Bx  Ch�                                    Bxn�B  �          @�=q�I��@6ff@FffB�\C	���I��?�@p  B5�RC�)                                    Bxn���  
�          @��R����@8��@�Q�BT��B������?�z�@�(�B���C.                                    Bxn�Ў  T          @��׿��@`��@s33B.�RB�p����@�@��HBb\)C c�                                    Bxn��4  <          @�  ��Q�@W
=@tz�B2�B����Q�@�R@�=qBd\)C{                                    Bxn���  �          @��׿��@*=q@��BU
=B�LͿ��?�@��B��)C�                                    Bxn���  "          @�p��tz�?�ff?�p�A�G�C���tz�?xQ�@z�A�C%Ǯ                                    Bxn�&  "          @�����=�G��P����C2�{���?u�G��
�C'��                                    Bxn��  �          @������\>8Q��Z�H�
=C1�����\?�=q�P  �33C&                                    Bxn�(r  
Z          @�ff�i����Q������?  C5aH�i��?xQ��y���7�RC%
                                    Bxn�7  
�          @�  �K�?��@l��B5�C���K�?@  @�G�BL(�C&�3                                    Bxn�E�  T          @��
���\?��@B�\B�C"h����\>�ff@QG�B=qC-��                                    Bxn�Td  �          @�Q��Z=q>\@���BM��C-���Z=q�B�\@�  BJffC@�
                                    Bxn�c
  �          @�=q�@��?�R@�(�BaC(ff�@�׿(�@�(�BaC?�\                                    Bxn�q�  �          @�=q�4z�?:�H@�\)Bip�C%xR�4z��@�Q�Bk�\C>�H                                    Bxn�V  �          @��H�B�\?h��@��\B\��C#J=�B�\���R@��Bb�
C9޸                                    Bxn��  �          @�33�A�?G�@�(�B_��C%}q�A녾�ff@�p�Bb�C<k�                                    Bxn흢  �          @��H�U�?@  @��BO�C'c��U���
=@���BR��C;�                                    Bxn��H  
�          @��\�C33?333@�=qB^�C')�C33��@��HB`G�C=��                                    Bxn���  T          @�p���G�?��@Dz�B  C%����G�>��R@P��Bz�C0&f                                    Bxn�ɔ  
�          @�\)��ff?�G�@<��A�C%���ff>�ff@K�B�C.��                                    Bxn��:  
          @�  ���
?z�H@QG�B
��C({���
=�Q�@Z=qB�RC2�{                                    Bxn���  9          @�{��{?z�@Y��BC,����{���
@[�Bz�C8!H                                    Bxn���  T          @�(����?�
=@�A�ffC!����?��@"�\AָRC'ٚ                                    Bxn�,  
�          @��
���
?�p�?�33A��RC&u����
?333@
=qA�
=C,5�                                    Bxn��  k          @����?�\)?�z�A��C%aH���?c�
?���A�G�C*T{                                    Bxn�!x  
�          @�ff���H?�33?�33A�z�C"����H?���@��A��C'�=                                    Bxn�0  T          @�\)���?���@333A��C
���?���@K�B��C&�                                    Bxn�>�  T          @�ff��{?�Q�@!G�A�33C����{?�(�@;�A�\)C%aH                                    Bxn�Mj  T          @�{���\?�(�?�G�Aw�C'\���\?J=q?��
A�{C+��                                    Bxn�\  
�          @�{��
=?�Q�?��A8z�C�\��
=?˅?���A��\C#�                                    Bxn�j�  
�          @�
=��  @�\>�(�@�33C��  ?�{?p��A��C xR                                    Bxn�y\  
�          @�ff����@�?��
AQC�f����?��
?�ffA�ffC ��                                    Bxn�  T          @�  ��ff?У�>�\)@6ffC#aH��ff?�  ?0��@޸RC$�)                                    Bxn  
�          @�������@   ����'
=C������@   >��@#�
C�f                                    Bxn�N  "          @����ff@�R�aG����C��ff@Q쾏\)�1G�CaH                                    Bxn��  �          @�����
@p��G����C^����
@%������CL�                                    Bxn�  "          @�ff���@&ff���
�QC�)���@5�z����
C�3                                    Bxn��@  "          @������@7
=��z��b�HCB����@HQ�#�
��33C��                                    Bxn���  �          @����Q�@0  �У����\C  ��Q�@Dz�aG����C&f                                    Bxn��  �          @����@4z���
����CE���@K���  �#�C�                                    Bxn��2  �          @����\)@@�׿�
=��=qC{��\)@U�\(��  CJ=                                    Bxn��  T          @�����@:=q��Q���\)C�����@S�
����:�HC&f                                    Bxn�~  
�          @�z���ff@:=q�Q���C8R��ff@Z�H�Ǯ����C�\                                    Bxn�)$  �          @���|(�@=p��/\)��\C��|(�@c�
��33���C	��                                    Bxn�7�  
�          @�(���z�@0  �(���߮Cp���z�@Tz��{��(�C:�                                    Bxn�Fp  �          @����z�@G
=��33��=qC�f��z�@[��O\)�  C                                      Bxn�U  �          @���
=@N{����R=qC0���
=@\�;�(����
C^�                                    Bxn�c�  T          @���=q@=p��˅��=qC
=��=q@QG��J=q� Q�Cn                                    Bxn�rb  T          @����H@33�&ff����C�f���H@(���   ��p�C�                                    Bxn�  �          @�p���{@	���333���C(���{@1��
�H��C��                                    Bxn  �          @�p���
=?���K��	�C#J=��
=@ff�/\)��Q�C�
                                    Bxn�T  �          @�(���p�?p���"�\��\)C)8R��p�?�ff��R����C"��                                    Bxn��  
�          @��
���
@K���  �x��C����
@]p��(���ָRC�f                                    Bxnﻠ  "          @�=q���?�
=��ff��Q�C$aH���?�\����>�HC ޸                                    Bxn��F  �          @�Q���  ?�G�������C#�3��  ?�{�W
=�	��C"�                                    Bxn���  "          @�����  ?�\)>�@��C"޸��  ?���?Y��AffC$��                                    Bxn��  o          @�����@�\>\@z�HCQ����@�?p��A�C�                                    Bxn��8  �          @�����\?��H��
��z�Cٚ���\@�H��  �x��Cff                                    Bxn��  
(          @��\����@z�� �����C}q����@0  ��{�b�HCk�                                    Bxn��  T          @�����  @1G�� ������C����  @K���  �TQ�C+�                                    Bxn�"*  �          @�����  @;��p�����C�=��  @\�Ϳ�33����C8R                                    Bxn�0�  T          @��R�s�
@K��
=q��  C#��s�
@g�����]��CxR                                    Bxn�?v  �          @�Q���ff@#33��33���\C#���ff@<�Ϳ����K\)Cz�                                    Bxn�N  �          @�p��Z=q@S33�G
=���C���Z=q@~{�(����C�)                                    Bxn�\�  
�          @�33�^{@HQ��E��ffC	���^{@s33�����G�Cs3                                    Bxn�kh  "          @����hQ�@Mp��:=q��=qC
}q�hQ�@u��G���Ch�                                    Bxn�z  �          @�(��tz�@:�H�>{��  C���tz�@c�
�
=q��\)C��                                    Bxn���  "          @��\�p��@0  ���
���C�f�p��@Fff����Ap�C��                                    Bxn�Z  T          @����n�R?���@vffB2z�C#5��n�R>\)@�Q�B<
=C1�                                    Bxn�   �          @��j=q?��\@vffB5��C$k��j=q<�@\)B=C3�                                     Bxn�  T          @�����
?�Q�@,(�A�p�C �����
?8Q�@>{BG�C*\                                    Bxn��L  "          @�Q���\)?�33?Tz�A��C{��\)?��?��
AeC ��                                    Bxn���  "          @�������?�Q�?+�@��C�f����?��H?���AH��C E                                    Bxn���  "          @������@%�>��@:�HC
���@(�?c�
A
=C��                                    Bxn��>  
�          @�33��{?��?�Q�A��C$Y���{?k�?�(�A��\C(�3                                    Bxn���            @��H��p�>L��@%�A���C1�H��p���G�@#33A�(�C9)                                    Bxn��  T          @�����?�(�?!G�@���C ����?�G�?��A5�C#                                      Bxn�0  T          @�=q���@�?�R@�p�C5����?�z�?�\)A;�
CW
                                    Bxn�)�  �          @������H@����H����C\)���H@Q�=���?}p�C�f                                    Bxn�8|  T          @�(���Q�@<�;���ffC(���Q�@>�R>��@(��C��                                    Bxn�G"  �          @�����33@0�׿\(���CaH��33@8Q����\CE                                    Bxn�U�  �          @�{����@%��\)��C�\����@(��=�Q�?n{CG�                                    Bxn�dn  
�          @�����
=@'
=�
=q���RCQ���
=@*=q=�?��RCٚ                                    Bxn�s  �          @�(����@7�����<  Cff���@C�
������=qC�R                                    Bxn�  
�          @���@G
=�Y���	�C^���@N{�u���Ch�                                    Bxn�`  
�          @�\)��  @W�����*=qC#���  @a녾B�\��Q�C��                                    Bxn�  "          @�
=��{@Q�\�w�
C:���{@+��c�
��CxR                                    Bxn�  �          @��R��  @33�Q����C0���  @0  ��  �s�
C�                                    Bxn�R  �          @�{��  @Q��G����RC�f��  @'���
=��C#�                                    Bxn���  T          @�{��33@$z������\C����33@@  ����b�HC�H                                    Bxn�ٞ  T          @��R��\)@O\)����b{C���\)@_\)������C�                                    Bxn��D  o          @�p���z�@H�ÿ�  ���Ck���z�@^�R�n{�=qC��                                    Bxn���  9          @�ff���@K���G���Q�Cٚ���@aG��k����C
                                    Bxn��  "          @�\)���H@7
=��ff���C����H@Mp���ff�)G�C�q                                    Bxn�6  �          @�����H@Z�H���p�C&f���H@u���Q��@  C�                                    Bxn�"�  �          @�  ��z�@I����
=��p�CaH��z�@aG������1G�CB�                                    Bxn�1�  
�          @�ff�|(�@mp���\����C�3�|(�@��ÿL���CT{                                    Bxn�@(  "          @��R��{@a녿�����C޸��{@s�
�(����33C	��                                    Bxn�N�  T          @�ff���R@!녿�p��G�C�����R@0�׿z���G�C�                                    Bxn�]t  "          @�ff���@4zῐ���7�C�{���@@�׾����33C�                                    Bxn�l  o          @�(���\)@p��Q���RC�R��\)@%�B�\���RC�\                                    Bxn�z�  9          @������
@!G������X��C�����
@1G��(����Q�Cff                                    Bxn�f  
�          @�������@#33�У���ffCE����@7��s33�ffCQ�                                    Bxn�  
(          @�����(�@\(�?J=qA�C�f��(�@H��?�{A��RCc�                                    Bxn�  T          @�{���R@^{=#�
>�Q�C#����R@Vff?aG�A�
C�                                    Bxn�X  "          @������H@9����Q��n{C�����H@J�H�0���ᙚCc�                                    Bxn���  �          @�����@HQ������C(����@K�>B�\?��C��                                    Bxn�Ҥ  
�          @�  ��z�@(�ÿ�(��l��C�{��z�@:�H�G�����C�                                    Bxn��J  �          @�G���
=?�33���
�u�C ���
=@p����\�"�RC�                                    Bxn���  "          @�����=q@(������=qC����=q@*=q�˅��33C�                                    Bxn���  �          @�����\)@G��
�H��(�C�f��\)@\)��{��G�Ck�                                    Bxn�<  �          @������\@G������ffCT{���\@"�\��=q��  CB�                                   Bxn��  "          @����G�?�
=�   ����C���G�@�R������\)C�)                                   Bxn�*�  �          @�  ��?O\)�333��{C*����?�(��!G���z�C#\)                                    Bxn�9.  T          @�{����>�G��@�����
C.������?��3�
���C&Y�                                    Bxn�G�  �          @�
=��
=@�\�����{C���
=@(������W
=C޸                                    Bxn�Vz  
�          @�p���p�?������H��(�C&�f��p�?�녿�{��  C"Y�                                    Bxn�e   "          @�p����H?�
=�Q����C&�R���H?�z���
����C!�                                    Bxn�s�  �          @�p�����?�33���
�S
=C%0�����?��fff��
C"z�                                    Bxn�l  �          @�\)��
=>���ff�|z�C/+���
=?Tz῱��a�C+h�                                    Bxn�  T          @�  ��ff?�z��(Q���ffC&ٚ��ff?�G������Q�C h�                                    Bxn�  
�          @�Q���z�?��H�33��Q�C&Ǯ��z�?�p���Q���33C!^�                                    Bxn�^  T          @�G���?0�׿�\)���HC,����?��׿����{C(=q                                    Bxn�  T          @�Q�����?G���=q��z�C+����?��H�˅��  C'h�                                    Bxn�˪  "          @������?
=q��{�\��C.p�����?Y����Q��@Q�C+5�                                    Bxn��P  T          @�\)��(�?�
=��ff�|  C'�{��(�?��
��(��Dz�C$#�                                    Bxn���  =          @�  ��Q�?z��  �rffC-����Q�?n{����S�C*n                                    Bxn���  �          @��R����>�Q쿧��TQ�C08R����?+���
=�?33C-\                                    Bxn�B  T          @�
=���H>k���
=�=�C1�����H?�\��=q�.�HC.Ǯ                                    Bxn��  
�          @��R���=�Q���
�x��C3\���>����H�m��C/33                                    Bxn�#�  �          @�\)����>����(��mG�C2s3����?   ����`  C.�\                                    Bxn�24  �          @�
=��=L�;�z��9��C3u���=���=q�+�C2�                                    Bxn�@�  
�          @�
=��녽����R�IC533���>B�\��p��H(�C2�                                    Bxn�O�  
�          @�{��=q�(��p����C:G���=q��Q쿈���,��C7��                                    Bxn�^&  o          @�ff���R�#�
���ip�C4n���R>��R����c�
C0�                                    Bxn�l�  �          @�
=��  ��p��	����  C8���  >.{����(�C2&f                                    Bxn�{r  T          @��R��  ����z��e��C8�3��  �����R�q�C55�                                    Bxn�  "          @�ff��p��z���\��G�C:ff��p��u����ffC4��                                    Bxn���  "          @���\)��\)�-p���  C4���\)?���(����  C-+�                                    Bxn��d  �          @�����=q>��
�8�����C00���=q?��\�.{��p�C(�                                    Bxn��
  �          @�{��Q쾣�
��\��33C7� ��Q�>B�\�33����C1�R                                    Bxn�İ  
�          @������\���ͿTz��z�CD�R���\���Ϳ����E�CB33                                    Bxn��V  
�          @�\)��  �
�H�u�#�
CJk���  ��
=��\)CI��                                    Bxn���  �          @�z���G���33�#�
��\)CG�=��G�����R�ʏ\CF�)                                    Bxn��  
�          @�
=���H���R>���@���CHu����H�녽��Ϳ}p�CH�\                                    Bxn��H  �          @�  ���R��{?&ff@У�CDxR���R��(�>u@ffCE��                                    Bxn��  "          @�����p���?�@��CG� ��p��   =��
?Q�CHB�                                    Bxn��  �          @������R�z�?@  @�CL����R�(�>#�
?�=qCM
=                                    Bxn�+:  8          @�G���G��S33?�\)A�=qCY����G��i��?�  A!G�C\c�                                    Bxn�9�  "          @�G��c33�g�@.{AᙚCa���c33��?�  A��Ce��                                    Bxn�H�  �          @����Q��xQ�@.{A���Ce���Q���?�
=A��Ci�                                     Bxn�W,  T          @�p��~{�Z=q?��HA���C\�3�~{�r�\?���A0(�C_��                                    Bxn�e�  �          @��H��ff?(���Q���
=C-E��ff?��ÿ�p����RC(Y�                                    Bxn�tx  T          @����ff>�33�����C00���ff?O\)��\����C+.                                    Bxn��  "          @�����?
=��R����C-\)���?�\)�G����RC'�{                                    Bxn���  �          @�����H?G��(�����C+B����H?�ff��
=���
C%�3                                    Bxn��j  �          @�(���{?�\)��{��Q�C'ٚ��{?�����
�|  C#xR                                    Bxn��  T          @��H���?�33���
�}�C'�����?��R�����F=qC$&f                                    Bxn���  
�          @��H��z�?�G����H�s�
C#�H��z�?��ÿ���-G�C z�                                    Bxn��\  T          @��R����?��H��z��ep�C'ff����?\�����-p�C$L�                                    Bxn��  �          @�
=����?\)��{�3\)C.@ ����?O\)�n{�\)C+��                                    Bxn��  >          @�ff��?���G��w33C.n��?aG���=q�Y��C*�q                                    Bxn��N  j          @������ÿ#�
�k��{C:�����þ��Ϳ�ff�+�C8)                                    Bxn��  
�          @�33����?+���Q���{C,������?��׿�(���Q�C'�)                                    Bxn��  �          @�33���>\��Q����\C/����?G������{C+�H                                    Bxn�$@  
�          @�33���\>�p���{���
C0  ���\?B�\��(��s�C+��                                    Bxn�2�  T          @�G���  ���������z�C7xR��  >���G����HC0�R                                    Bxn�A�  p          @�33��
=>8Q쿔z��?�C2!H��
=>���=q�2=qC/0�                                    Bxn�P2  8          @�33����>�=q�G�� ��C1(�����>��0����33C/@                                     Bxn�^�  
�          @�(����=#�
�Q���\C3�H���>u�J=q� ��C1��                                    Bxn�m~  "          @�=q��  >u�:�H��C1}q��  >�녿&ff��{C/��                                    Bxn�|$  
�          @�����\)>���ff��ffC/33��\)?\)���
�W�C.0�                                    Bxn���  T          @�(����>��Ϳ�G���{C/�����?Tz��������C+#�                                    Bxn��p  "          @����  >.{����Z�RC20���  ?   ��  �L��C.��                                    Bxn��  �          @���>.{��
=����C2.��?z�˅����C-�)                                    Bxn���  "          @�����>���\�z=qC1B���?!G���z��g�C-c�                                    Bxn��b  
�          @�����z�>aG���33���C1����z�?!G�����}��C-^�                                    Bxn��  �          @�33���\>��
��z���ffC0�=���\?:�H���
�~=qC,@                                     Bxn��  T          @����p�>�
=�\)��Q�C/{��p�?��
��
��p�C(+�                                    Bxn��T  �          @��\��33>aG���R���\C1z���33?E��ff��\)C+ff                                    Bxn���  T          @�=q��{>��Ϳ�
=���HC/����{?^�R��\����C*�=                                    Bxn��  	�          @�z����>����  ���C/�\���?s33�����HC)��                                    Bxn�F  	          @�p���Q�>���   ��=qC/�
��Q�?c�
����{C*aH                                    Bxn�+�  �          @�z�����?n{��{��p�C*
����?�ff��=q�[33C&:�                                    Bxn�:�  �          @�p���(�?
=�\)��p�C-n��(�?����G���=qC'�\                                    Bxn�I8  �          @�p�����?��\��\����C(�{����?�  ��(���p�C#u�                                    Bxn�W�  �          @����z�?\�+����C$8R��z�?�녾�=q�.{C#\                                    Bxn�f�  T          @�
=��(�?�\�����(�C!�q��(�?���=L��>�C!L�                                    Bxn�u*  �          @����  @p���=q��  C�f��  @1G��k���C�3                                    Bxn���  �          @����ff@  ��p��s
=C�=��ff@#33�^�R�G�C�q                                    Bxn��v  �          @�����z�@G���{����C���z�@�����T��C�H                                    Bxn��  �          @��H��z�@G������G�C����z�@����(��JffC��                                    Bxn���  �          @�����@Q��\)��ffC�f���@.{�xQ��\)C��                                    Bxn��h  �          @�33���@'��������C�����@<�Ϳn{�p�C��                                    Bxn��  T          @��
��ff@7���{�5�C����ff@C�
��33�e�C�R                                    Bxn�۴  T          @����@1녿�33�;33C����@>�R�����p�C0�                                   Bxn��Z  �          @�ff���@0�׿��R�IG�C.���@>�R�   ���\C8R                                    Bxn��   T          @�����@   ��{����C{��@4z�k���C(�                                    Bxn��  �          @�G���
=@�׿�ff��ffC�\��
=@(�ÿ�
=�<��C                                    Bxn�L  T          @�������@Q��{��
=C������@7
=��ff�x��C
=                                    Bxn�$�  T          @����\)@�����p�C8R��\)@333��
=��C^�                                    Bxn�3�  �          @�Q���ff@zῬ���Y�CT{��ff@�G���=qC                                    Bxn�B>  �          @�Q����@33�k��p�C���@p����R�G
=CL�                                    Bxn�P�  �          @������R@p���(���G�C���R@(Q쿫��W
=C
=                                    Bxn�_�  �          @�G����H@33�	����33C�����H@1G���  �p��C0�                                    Bxn�n0  �          @����p�?��
�����F�\C!
=��p�@G��5���C�H                                    Bxn�|�  �          @�ff��ff?�  �:=q��ffC����ff@�����=qCaH                                    Bxn��|  �          @�p�����?�G��}p��/\)C�����?�����\��(�C�f                                    Bxn��"  �          @���(�@�\>�
=@�=qC����(�@
=?}p�A"�\C��                                    Bxn���  �          @��R���@  ?c�
A��C=q���?���?�
=Ak
=C!H                                    Bxn��n  �          @������
@�R?�Q�Ag
=Cz����
?��?��HA��HC                                     Bxn��  �          @�33��  @G�?���A2ffC�)��  ?�z�?�A�33C �                                    Bxn�Ժ  �          @��
��p�@�
=����C����p�@
�H�����
C(�                                    Bxn��`  �          @�������@���z��]�C5�����@=q�Q�� (�C�\                                    Bxn��  j          @�{��  @(����
�q��Cc���  @ �׿k��Q�Cz�                                    Bxn� �  �          @�p����@��\���C�����@#33��
=�a�C��                                    Bxn�R  �          @�ff���@�R�\(��G�C����@'��B�\��{C�R                                    Bxn��  �          @�����H?�z������RC L����H@���=q�M�Ck�                                    Bxn�,�  �          @������?��H�333���C.����@'������G�C��                                    Bxn�;D  �          @�
=���?˅�E��
=C!�3���@��%���C�                                    Bxn�I�  �          @�\)����?�p��w��"(�C �H����@��W
=�	G�Cc�                                    Bxn�X�  �          @��R����?����(��0
=C!�����@=q�hQ���CG�                                    Bxn�g6  �          @��p  ?�\)��{�B=qC#u��p  @{��  �)��C\)                                    Bxn�u�  �          @�{�l��?�(���ff�C=qC!Ǯ�l��@��\)�)ffC��                                    Bxn���  �          @�{���?���u�� ��C"&f���@��U�	(�C��                                    Bxn��(  �          @�����?�33�s�
� p�C!�����@ff�Tz��p�Ch�                                    Bxn���  
�          @�z��~�R?�(���p��433C"��~�R@���mp��{C^�                                    Bxn��t  �          @��
�\)?���(��3(�C#�3�\)@���l(��C#�                                    Bxn��  �          @���o\)?�  ��(��B  C$�q�o\)@
=�~{�*�HC�)                                    Bxn���  �          @��H�c33?xQ������KQ�C$�3�c33@
=����3\)C8R                                    Bxn��f  �          @����E?u��=q�`p�C"���E@������E=qC�                                     Bxn��  �          @�Q��:=q?Tz������i(�C$0��:=q@������Np�C��                                    Bxn���  �          @�=q�x��?��H�   �хC"��x��?�Q��\)���C�=                                    Bxn�X  �          @��
���@{>Ǯ@uC�����@�\?uA�RC\)                                    Bxn��  �          @��
��@�?��@�z�C���?��?��A2�RC �{                                    Bxn�%�  �          @�\)����?��Ϳ&ff��
=C!�
����?��H�\)���C �H                                    Bxn�4J  �          @�  ����?����L�Ϳ��C �3����?�
=>��
@EC!�                                    Bxn�B�  �          @�����33@   ?O\)@��C W
��33?��H?��AJ{C"��                                    Bxn�Q�  �          @�  ��G�@33?�=qA&=qC����G�?�Q�?�=qAt��C"�q                                    Bxn�`<  
�          @�����33?�p�?���A(  C ����33?�\)?˅As\)C#ٚ                                    Bxn�n�  �          @�G���(�?���?���A7
=C"#���(�?�Q�?У�A{33C%��                                    Bxn�}�  �          @�G����?\?��HA`z�C$����?�=q?���A��
C).                                    Bxn��.  �          @�����(�?��H?˅At  C%n��(�?}p�?�A�Q�C*                                    Bxn���  �          @����  ?�{?޸RA��C#�R��  ?��@ffA�z�C(�\                                    Bxn��z  �          @�{���?�ff?�z�A��C#�q���?}p�@  A�  C)��                                    Bxn��   �          @�p���?Ǯ?�G�A��RC#����?��@�A��C)0�                                    Bxn���  �          @�p���p�?��?��A��\C$#���p�?�G�@Q�A���C)s3                                    Bxn��l  �          @������?��?�  A�C&�����?E�@�A��RC+�R                                    Bxn��  �          @�z����?�33@��A�(�C%33���?G�@(�A�33C+��                                    Bxn��  �          @�z���33?���@
=qA�(�C'O\��33?z�@��A��RC-�                                    Bxn�^  >          @�z���(�?�ff@
=qA��
C)���(�>�(�@
=A���C/n                                    Bxn�  j          @�(���(�>��
@�A�\)C0�H��(�����@�A��C7=q                                    Bxn��  �          @��\��������@�HA��HCC��������?�(�A�  CI(�                                    Bxn�-P  �          @�������?G�@\)A�ffC+Y�����=�Q�@'
=A���C3
=                                    Bxn�;�  �          @�������>��H@$z�A�G�C.����;W
=@'
=AָRC6c�                                    Bxn�J�  �          @�����z�>�  @(Q�A��HC1@ ��z���@&ffA�(�C9=q                                    Bxn�YB  �          @�����녽�@0��A���C5ff��녿\(�@(Q�A���C=��                                    Bxn�g�  �          @�G����\=L��@0��A�(�C3z����\�333@*�HA�{C;ٚ                                    Bxn�v�  �          @�Q���
=>��R@9��A�C0h���
=��@7�A�\)C9��                                    Bxn��4  �          @�
=�����@<��A�(�C5n����k�@3�
A���C>��                                    Bxn���  �          @�\)����>�
=@<(�A���C/\���;Ǯ@<��A�p�C8�=                                    Bxn���  �          @�  ��  ?
=@K�B�\C,����  ����@N{Bz�C7�)                                    Bxn��&  �          @��H��G�?���@g
=B  C����G�?\)@z=qB(\)C,��                                    Bxn���  �          @��
��=q?���@n�RB{C"����=q>�\)@}p�B)��C0G�                                    Bxn��r  �          @��
��(�?�\)@j=qB=qC"����(�>��
@x��B&(�C/ٚ                                    Bxn��  �          @��
���
?��H@mp�B{C$z����
>��@y��B&�HC1�R                                    Bxn��  �          @�(����\?�=q@s�
B"  C%�����\    @}p�B)�HC4�                                    Bxn��d  T          @�(�����?aG�@|(�B({C(n���þu@���B,�
C7#�                                    Bxn�	
  �          @�z���Q�?W
=@~�RB*�C(�)��Q쾏\)@��B.=qC7�{                                    Bxn��  �          @��\����?G�@���B.ffC)ff���;�Q�@��\B1��C8�                                    Bxn�&V  T          @��H��\)?G�@|��B*Q�C)����\)��{@�Q�B-��C8��                                    Bxn�4�  �          @�=q��?:�H@~�RB,�\C*)���Ǯ@���B/�C9c�                                    Bxn�C�  �          @��H��(�?B�\@�=qB0Q�C)�
��(��Ǯ@��
B3(�C9k�                                    Bxn�RH  �          @�(�����?+�@��RB7{C*�)������@�\)B8G�C;T{                                    Bxn�`�  �          @��
���\?
=@�B5�\C+�\���\�
=@�B5�C<@                                     Bxn�o�  �          @�(��~�R>�(�@���B<
=C-���~�R�E�@�  B9G�C?�                                    Bxn�~:  �          @���u>��R@�\)BE=qC/c��u�s33@�z�B?��CA�
                                    Bxn���  �          @����u>u@�
=BD�C0xR�u��G�@��B>z�CB�                                    Bxn���  �          @�33�mp�>W
=@�Q�BI��C0�3�mp���ff@�z�BB��CC�=                                    Bxn��,  �          @���qG�>�Q�@��BEQ�C.���qG��aG�@��\B@�HCA+�                                    Bxn���  �          @�(��x��>��@�(�B@C-
=�x�ÿE�@��HB>\)C?(�                                    Bxn��x  �          @���l(�?L��@�
=BGG�C'�l(���@���BJ{C;p�                                    Bxn��  �          @�=q�e?u@��RBI  C%{�e����@���BN��C9J=                                    Bxn���  T          @��\�Tz�?�@���BSp�C ���Tz�8Q�@�G�B\�HC7+�                                    Bxn��j  �          @�=q�<(�?J=q@��RBi�RC$���<(��!G�@�\)Bkp�C@&f                                    Bxn�  �          @��\�Z=q?��\@��BQ{C#L��Z=q���R@��RBWC9=q                                    Bxn��  �          @�33�o\)?fff@��BC�C&� �o\)�\@�\)BH(�C9�{                                    Bxn�\  T          @����  ?!G�@�p�B7�\C+���  ��@�B8  C<+�                                    Bxn�.  �          @�33���?J=q@�z�B4{C)��녾��@�{B7{C9��                                    Bxn�<�  T          @����HQ�?Ǯ@��BPG�C���HQ�>k�@�  Bb{C/��                                    Bxn�KN  �          @��\�K�?�p�@���BM�Cp��K�>\@��\Ba��C-0�                                    Bxn�Y�  �          @��\�Tz�?�z�@�
=BI{Ch��Tz�>��
@�  B[��C.h�                                    Bxn�h�  �          @���b�\?��@�B8�RC\�b�\?(�@�G�BNz�C*.                                    Bxn�w@  �          @�G��U?���@���BN  C�q�U    @�
=B[  C3�                                    Bxn���  �          @�33�Q�?���@��HBNp�CY��Q�>L��@��HB_�C0��                                    Bxn���  �          @�(��HQ�?��@�ffBTQ�C^��HQ�>k�@�
=Bg
=C/޸                                    Bxn��2  T          @�(��H��?�@�BR�C�q�H��>��@��RBf=qC/:�                                    Bxn���  �          @�z��4z�@�
@�\)BVG�C�=�4z�?�R@�(�Brz�C'��                                    Bxn��~  �          @�� ��@ff@��\BZp�C��� ��?\(�@���B}�
C!�                                    Bxn��$  �          @��,��@{@�G�BWffC�
�,��?@  @�
=Bw{C$�                                     Bxn���  �          @�ff�,��@�R@�=qBX{Cs3�,��?=p�@�  Bw�HC$�\                                    Bxn��p  i          @�
=�4z�@@�=qBW��Ch��4z�?(�@�
=Bt\)C'                                    Bxn��  T          @���#33@�R@�B^(�C
޸�#33?5@��
B  C$�                                    Bxn�	�  �          @���<(�?���@��BV��CxR�<(�>��@�p�Bp
=C*�
                                    Bxn�b  �          @�ff�<��?�\)@��\B\\)C33�<��>��@��HBop�C1�                                    Bxn�'  �          @�
=�w
=?��
@��
B<33C!�R�w
=�L��@���BFC4�R                                    Bxn�5�  �          @�\)�`��?���@�(�BK�RC��`�׽L��@��\BW�HC4�H                                    Bxn�DT  �          @���O\)?��
@��BY��Cc��O\)�B�\@���Bd�C7^�                                    Bxn�R�  �          @���N�R?�\)@���BSG�CaH�N�R>#�
@���Be
=C133                                    Bxn�a�  �          @�\)�dz�?��@�  BC�\CT{�dz�>�  @���BT�
C/�R                                    Bxn�pF  �          @�
=�^{?��H@�{BO�RC ���^{�W
=@��HBYQ�C7�                                    Bxn�~�  �          @���k�?Q�@�z�BK�RC's3�k��z�@�BM�C<�3                                    Bxn���  �          @�  �\��?(�@��
BX�C)�f�\�ͿW
=@��\BV�RCA��                                    Bxn��8  �          @�ff�Q�?���@�  BTffC#��Q녽�\)@�{Ba��C5L�                                    Bxn���  �          @�ff�_\)?u@��RBQ��C$�
�_\)���@�G�BV�\C;��                                    Bxn���  �          @�{�]p�?h��@��BS�C%@ �]p���@���BW��C<��                                    Bxn��*  �          @�{�s�
?p��@��BA\)C&0��s�
���@��BF{C:&f                                    Bxn���  �          @�p��W
=?z�@��\B[  C*(��W
=�^�R@�G�BX
=CB��                                    Bxn��v  �          @�p��[�?(��@���BW\)C)��[��J=q@�Q�BV�C@��                                    Bxn��  �          @�{�U?B�\@��B[=qC'+��U�8Q�@��
B[�C@#�                                    Bxn��  �          @�
=�R�\?333@�B^C'�R�R�\�L��@�p�B]�RCA�f                                    Bxn�h  �          @�ff�U�?@  @��
B[��C'L��U��=p�@�(�B\{C@}q                                    Bxn�   �          @�p��<(�?fff@���Bj�
C#�<(��&ff@��HBm��C@��                                    Bxn�.�  �          @�ff�HQ�?�ff@�ffBa\)C!}q�HQ��@�G�Bgz�C<��                                   Bxn�=Z  �          @�{�O\)?z�H@�(�B\��C#+��O\)��@�ffBa�RC=E                                   Bxn�L   �          @�p��R�\?���@�G�BX33C!���R�\�Ǯ@���B_��C:�                                     Bxn�Z�  �          @��e�?�  @��
BL�
C$xR�e���G�@��RBR(�C;�                                    Bxn�iL  �          @�
=�p��?k�@�G�BFQ�C&E�p�׾��H@��BJ=qC;}q                                    Bxn�w�  �          @�
=�dz�?�z�@�z�BL(�C!��dzᾔz�@���BT��C8��                                    Bxn���  �          @�\)�hQ�?��
@�z�BKp�C$)�hQ��
=@�\)BQQ�C:��                                    Bxn��>  �          @�
=�q�?���@�p�B>�HC Y��q녽L��@��
BJ�RC4�=                                    Bxn���  �          @�
=�`  ?�@�(�BK�C�
�`  �L��@��HBX��C4Ǯ                                    Bxn���  �          @����b�\?�p�@�z�BI�CE�b�\    @��
BW�HC3�3                                    Bxn��0  �          @�G��p��?L��@�z�BI��C(�p�׿(��@�p�BJ�C=��                                    Bxn���  �          @����e�?aG�@�  BP�
C&B��e���R@�G�BS\)C=Ǯ                                    Bxn��|  �          @���]p�?�=q@�  BR�C"���]p���
=@��BY33C:�3                                    Bxn��"  �          @�Q��G�?�G�@��
BZffC5��G��u@�33Bj=qC5{                                    Bxn���  �          @�  �6ff?�Q�@��Ba�CQ��6ff=��
@���Bv33C2c�                                    Bxo 
n  �          @��R�P��?��@��HBZ{C ��P�׾���@��RBb  C;�                                    Bxo   �          @�
=�^�R?���@�z�BLp�C^��^�R��@��HBY{C5�                                    Bxo '�  ?          @���^�R?�p�@��
BK  C�R�^�R���
@�33BY�\C4T{                                    Bxo 6`  i          @�  �c�
?u@�BO
=C$�H�c�
��@�  BSQ�C<}q                                    Bxo E  �          @����?\)?��H@��\BY�CG��?\)>�@��
Bn��C1��                                    Bxo S�  �          @����5@z�@�G�BR{C�R�5?8Q�@���Bs�C%��                                    Bxo bR  �          @����<��@\)@���BP��C�q�<��?&ff@��BpG�C'��                                    Bxo p�  �          @���'
=@�@�ffB[�C
:��'
=?+�@�B~�C%��                                    Bxo �  �          @�(��=p�@	��@��BTG�C
=�=p�?�@�G�Bq��C)��                                    Bxo �D  �          @����{�p��@��B7=qC@�)��{�G�@y��BG�CPs3                                    Bxo ��  �          @�(�����J=q@�B9(�C>�������@}p�B!  CO0�                                    Bxo ��  �          @�(�����E�@�\)B;�
C>������Q�@�Q�B#�\COs3                                    Bxo �6  �          @��
���ÿG�@���B?Q�C>�f�����
=q@��B&\)CP0�                                    Bxo ��  �          @Å��녿5@�Q�B>=qC=�����@���B&ffCOB�                                    Bxo ׂ  T          @\��Q��\@�G�BA�C;33��Q��z�@���B,
=CMu�                                    Bxo �(  �          @�33���þ��@��BA��C9�{���ÿ�=q@�ffB.  CLp�                                    Bxo ��  �          @��
��  ���R@��
BD\)C8n��  ��G�@�G�B2  CK��                                    Bxot  �          @\�w
=��\)@�{BJQ�C5��w
=��ff@�B;G�CI��                                    Bxo  �          @\�tz὏\)@�\)BLffC5\�tz����@�
=B=  CJW
                                    Bxo �  �          @����\�Ϳ��@���BY��C=�{�\����@�ffB>�RCS��                                    Bxo/f  �          @�33�Z�H�0��@�  B\Q�C?n�Z�H�  @���B?Q�CUO\                                    Bxo>  �          @��
�<�ͿE�@���Bp=qCB�3�<����H@���BM\)C[ff                                    BxoL�  �          @���mp���@���BS��C6\�mp���Q�@�33BA��CL��                                    Bxo[X  �          @�p��w
==�\)@�=qBM(�C2��w
=��(�@��\B?�CH��                                    Bxoi�  �          @�z��fff=���@�
=BXffC2z��fff��  @��BJ33CJ�3                                    Bxox�  �          @��
�g�>\)@�BV��C1�=�g�����@��RBIz�CI�
                                    Bxo�J  �          @�(��p  >.{@��HBP�
C1n�p  ��33@�z�BD��CHxR                                    Bxo��  �          @��
�j=q<��
@�z�BT�C3�f�j=q�Ǯ@�z�BEp�CK                                    Bxo��  �          @��
�e=L��@��RBXQ�C3!H�e��ff@��RBI(�CKY�                                    Bxo�<  �          @�33�e>L��@�p�BWffC0ٚ�e��z�@�
=BK
=CIc�                                    Bxo��  �          @�=q�g�>L��@��
BUffC0�f�g����@�p�BI33CI�                                    BxoЈ  �          @�33�fff>L��@��BV�HC0�{�fff��z�@��RBJ�CIY�                                    Bxo�.  �          @\�`��=��
@��RBZ�RC2���`�׿��@�
=BK��CK�)                                    Bxo��  �          @�33�_\)=�G�@�Q�B\�\C2��_\)�\@���BM�RCK�H                                    Bxo�z  �          @�z��Vff>#�
@�z�Bc��C1@ �Vff���
@��BT��CL��                                    Bxo   �          @�(��_\)>k�@���B\�RC0E�_\)��
=@��\BO��CJJ=                                    Bxo�  �          @����e�>�z�@��BY�C/T{�e���{@��BN  CH��                                    Bxo(l  �          @��
�l��>\@��HBQ�HC.!H�l�Ϳ�p�@�ffBI33CFff                                    Bxo7  
�          @��
��{?(��@��B8Q�C+��{�Tz�@�(�B6�
C?!H                                    BxoE�  �          @�ff��ff?p��@��RB+=qC(���ff�   @�G�B/  C:u�                                    BxoT^  �          @�p����?z�H@���B0�
C'=q��녿   @�(�B5{C:�
                                    Bxoc  �          @�(����?��\@�B8��C&#������\@�Q�B=�RC;�                                    Bxoq�  �          @Å�^�R?G�@�
=BYG�C'Q��^�R�h��@�{BW�CB�H                                    Bxo�P  �          @��H�~�R?�{@�(�B8��C!0��~�R�8Q�@��\BC��C6�                                    Bxo��  �          @�����(�@p�@#33A�(�CJ=��(�?�(�@N{B33C#B�                                    Bxo��  �          @�G���@#33@5A�ffCff��?��H@a�B  C"�                                    Bxo�B  �          @������@z�@j�HB(�CY����?8Q�@���B.(�C*u�                                    Bxo��  �          @�������@   @8��A�C������?��@c�
B  C#Y�                                    BxoɎ  �          @�Q����
@(�@<��A�C0����
?��@fffB\)C$33                                    Bxo�4  �          @�  ����?Ǯ@�z�B/�CǮ����=��
@�B>C2��                                    Bxo��  �          @�Q��u?��@�p�B=C!=q�u���@�33BHG�C7�{                                    Bxo��  �          @�  �j�H?��@���BE\)C�f�j�H��=q@�\)BP��C80�                                    Bxo&  �          @���z=q?�z�@���B+�C��z=q>�(�@��RBB=qC-��                                    Bxo�  �          @����z=q@(Q�@p  B{C\�z=q?��H@�p�B<�
C"�{                                    Bxo!r  �          @����fff?��@�{B>
=C^��fff>�=q@�=qBTz�C/�3                                    Bxo0  �          @�  �w
=@�
@��\B+�
C��w
=?��@���BE�C+�{                                    Bxo>�  �          @�ff��?�(�@qG�B��C����?�@��RB3�C,.                                    BxoMd  �          @��R��p�?�Q�@~{B'��C �f��p�<#�
@�\)B5�C3�                                    Bxo\
  �          @�  ���?O\)@���B/�C)8R����(��@�B1{C<�
                                    Bxoj�  �          @�  ���\?!G�@��B,��C+����\�Q�@�=qB+{C>�                                    BxoyV  �          @����z�H�u@���BD�C7u��z�H���
@�ffB1{CLs3                                    Bxo��  �          @�������=L��@�ffB1\)C3\)������33@}p�B$��CE�q                                    Bxo��  �          @�����
=>aG�@�G�B5��C1
=��
=���\@�33B+�RCD�R                                    Bxo�H  �          @�G���Q�?
=@�\)B2
=C,���Q�fff@�p�B/�C?�3                                    Bxo��  �          @\��p�����@�z�B,Q�C7�H��p����H@r�\Bp�CI(�                                    Bxo  
�          @\���>�@��B-33C2B�������@|��B"�\CD�\                                    Bxo�:  �          @Å����>�@�ffB-�C-����Ϳ�  @�33B)�C@�
                                    Bxo��  �          @Å��z�?&ff@�{B-\)C+���z�W
=@��B+��C>�\                                    Bxo�  �          @�����Q�?#�
@�\)B1��C+c���Q�^�R@�{B/�
C?}q                                    Bxo�,  �          @�����p�>L��@���B7\)C1E��p����@�33B,��CEp�                                    Bxo�  �          @����w
=��R@�ffBBC=&f�w
=�Q�@|��B'z�CP�                                   Bxox  �          @��H���\@c33?uA��C����\@?\)@��A��C:�                                   Bxo)  �          @��
��p�@[�?�
=A2=qC&f��p�@333@�
A�G�Cc�                                    Bxo7�  �          @�(���  @Q�?���AF�HC�q��  @'
=@��A�G�Cs3                                    BxoFj  �          @�������@B�\?��A�33C������@p�@/\)A�C@                                     BxoU  �          @��
���\@:=q?�A�G�C(����\@@.{Aң�C��                                    Bxoc�  �          @�(���
=@@��?�
=A��RC�\��
=@��@7
=A�
=C�q                                    Bxor\  �          @�z���=q@L��@�\A�
=Cff��=q@�@B�\A�G�C��                                    Bxo�  T          @Å��Q�@O\)@�A�p�C�R��Q�@z�@C33A�C�                                    Bxo��  �          @Å���@Mp�@ffA�\)C����@  @FffA��RC��                                    Bxo�N  �          @Å��  @C�
@33A��HCB���  @�\@N�RA��C�                                     Bxo��  �          @�(���33@.{@$z�A�(�C����33?У�@W
=B\)C!n                                    Bxo��  �          @�(���=q@"�\@2�\Aأ�C.��=q?���@`��B��C#��                                    Bxo�@  �          @����(�@+�@(��A��HC:���(�?Ǯ@Z�HB�C"B�                                    Bxo��  �          @�������@(��@4z�Aڏ\C�����?��H@e�Bp�C#�                                    Bxo�  �          @�z����H@"�\@333A���CY����H?�\)@aG�B  C$@                                     Bxo�2  �          @�������@�H@A�A��C33����?�@l(�B��C&G�                                    Bxo�  �          @�p���ff@   @G�A���C���ff?�(�@s�
B�C%}q                                    Bxo~  �          @�p����@�R@C�
A�(�CT{���?��H@p  B�C%�)                                    Bxo"$  �          @������@�\@I��A�z�CT{����?�  @qG�B�C(+�                                    Bxo0�  �          @����\)@{@C�
A�z�Cs3��\)?���@o\)B�C%�\                                    Bxo?p  T          @���{@   @H��A�Q�C���{?���@u�BffC%�3                                    BxoN  �          @�{��
=@
=@N�RA��Cff��
=?��
@w�B��C'��                                    Bxo\�  �          @�(����R@��@A�A홚C�����R?�
=@mp�B�\C%�                                    Bxokb  �          @��H��@p�@A�A��CJ=��?�
=@mp�BffC%�=                                    Bxoz  �          @�33��  @ ��@7�A��C!H��  ?��@eB=qC$�                                    Bxo��  �          @�33���@Q�@H��A��C����?���@r�\B��C'\                                    Bxo�T  �          @�������@�@_\)B�HC�=����?c�
@�33B,
=C(B�                                    Bxo��  �          @�����ff@@aG�Bz�C�
��ff?&ff@���B'��C+�f                                    Bxo��  �          @��H���?���@|��B!��C����=#�
@�Q�B1�\C3u�                                    Bxo�F  �          @����
=?�p�@���B-p�C#���
=�\@��B6�C9!H                                    Bxo��  �          @������\?���@w
=BC ���\=#�
@�p�B/��C3s3                                    Bxo��  �          @������?�33@tz�BQ�C=q���>�@��B.��C2Y�                                    Bxo�8  �          @�G���(�?��@w
=B  C �
��(�<#�
@��B.{C3�                                    Bxo��  T          @\���?���@��RB/33C%{�����\@��\B5z�C:ٚ                                    Bxo�  �          @�=q�E��aG�@��RBm\)C8��E����@�Q�BO  CV�                                     Bxo*  �          @�33�5��=q@�z�Bxp�C9aH�5�  @���BV�CZk�                                    Bxo)�  �          @�33�J=q��\)@�
=BkG�C5Y��J=q� ��@��BP�CT�                                    Bxo8v  �          @�33�E���=q@�Q�BnG�C9{�E��{@���BN=qCW�                                     BxoG  �          @��
�K����R@�
=Bj
=C9���K��\)@�
=BJ{CW(�                                    BxoU�  T          @�(��9����33@�(�Bu�HC:�)�9���@��BR  CZ�
                                    Bxodh  �          @�z��'
=�8Q�@�  B=qCC���'
=�/\)@��\BO�
Cb^�                                    Bxos  �          @\�E�����@�\)Bm��C5�\�E���
@��BQ��CU��                                    Bxo��  �          @�33�n�R?&ff@���BN�C*(��n�R����@�BH{CE�{                                    Bxo�Z  �          @�(��i��?
=@��\BR�C*���i�����
@�{BI��CGE                                    Bxo�   T          @����vff?O\)@�\)BH�C(&f�vff���@�{BFQ�CC�                                    Bxo��  T          @�(��tz�?(��@�
=BJ��C*&f�tz῕@��
BDCE
                                    Bxo�L  T          @���q�>�z�@��\BO�C/���q녿���@��\B@�CJ�                                     Bxo��  �          @��R�r�\?.{@�G�BF�RC)޸�r�\����@�ffBA�HCD!H                                    Bxo٘  �          @�
=�p  >��@�(�BK�\C-�=�p  ���@�{B@(�CHL�                                    Bxo�>  �          @����r�\>#�
@�Q�BG�HC1���r�\����@��B7�RCJ�                                     Bxo��  �          @�z��j�H=��
@�33BM�C2�R�j�H��
=@���B:�HCL��                                    Bxo�  �          @�(��X��>��@���BZG�C/���X�ÿ�=q@���BIp�CM�                                    Bxo0  �          @�Q��h��>�@�  BQQ�C,Ǯ�h�ÿ��@�=qBF{CH�3                                    Bxo"�  �          @�\)�q�?!G�@�=qBH  C*z��q녿�@��RBACE&f                                    Bxo1|  �          @�\)��
=?��@��HB,�C%����
=���@�B1C<\                                    Bxo@"  �          @�
=����?�=q@���B0{C%^����Ϳ��@��B5\)C<(�                                    BxoN�  T          @�{���H?���@z=qB$z�C%�R���H���H@���B*�HC:h�                                    Bxo]n  T          @�Q����?��H@S33B�
C%{��녾��@`��B\)C5�                                    Bxol  �          @�ff��33?��@E�BC#�q��33=L��@W
=Bp�C3ff                                    Bxoz�  �          @�  ��\)?���@>{A���C#����\)>�@QG�B
C2xR                                    Bxo�`  �          @�����=q@   ?�{A��\Cz���=q?�\)@\)A���C'�                                    Bxo�  �          @�\)��(�@��?���A1G�C5���(�?�z�?��A���C"�                                    Bxo��  �          @������R@G�?��@�G�Cp����R?�{?�Q�AiG�C ^�                                    Bxo�R  �          @������@+�>B�\?��CL����@��?�(�AC\)C�H                                    Bxo��  �          @�G�����@4z�>�?��CǮ����@#33?�p�AC�C:�                                    BxoҞ  �          @��H��=q@8Q������Cff��=q@,(�?��\A!G�C�                                    Bxo�D  �          @�(���
=@(Q�?��@�z�CL���
=@��?��Ar�RC&f                                    Bxo��  �          @�����Q�@6ff�\)��Q�CaH��Q�@5?(�@���CxR                                    Bxo��  �          @�(���Q�@B�\�Ǯ�u�CǮ��Q�@<��?Tz�A�RC�=                                    Bxo6  �          @������@5��L�����HC�R����@:�H>\@p��C�q                                    Bxo�  h          @������@3�
���\� (�C�����@>�R>.{?�Cp�                                    Bxo*�  �          @�33����@4z�B�\���C������@8��>�
=@��C�                                    Bxo9(  �          @�
=�|(�@I������θRCY��|(�@w
=��  �#�
C�{                                    BxoG�  �          @�G���G�@Y����=q��ffC����G�@o\)���Ϳ�  C
�                                    BxoVt  �          @�  �8Q�@HQ��w
=�(�C���8Q�@���33����B�=q                                    Bxoe  �          @�p��,(�@W��l���"��C �{�,(�@�=q��
���B���                                    Bxos�  �          @���,(�@hQ��Y�����B�(��,(�@�������B���                                    Bxo�f  �          @�
=�'�@`  �k�� z�B����'�@���p���(�B�z�                                    Bxo�  �          @�{�X��@G
=�6ff��=qC	}q�X��@\)��\)�j�\CaH                                    Bxo��  �          @�����\)@"�\��{�:�HC�q��\)@1G�    <#�
C�)                                    Bxo�X  �          @����(�@��W
=�
�HC���(�?��R?#�
@��C�=                                    Bxo��  T          @������?�z�Y���ffC"+����?�{��G���33C +�                                    Bxoˤ  �          @��R����@녿���2ffC������@   �#�
���CT{                                    Bxo�J  h          @��mp�@P�׿�\)��  C
��mp�@o\)������=qCǮ                                    Bxo��  �          @��
�l��@6ff��Q���\)Cc��l��@L(��.{���RC:�                                    Bxo��  �          @��\��{?�=q@Dz�B(�C#J=��{���
@VffB�C4O\                                    Bxo	<  �          @��\���R?�\@=qA�
=Cff���R?#�
@:�HA���C,@                                     Bxo	�  �          @�33��Q�?�@�\A���C�)��Q�?B�\@5A�C*�R                                    Bxo	#�  �          @�33����@ff@G�A��C:�����?���@,��A�
=C'�                                    Bxo	2.  �          @����  ?��@3�
A�\)C&xR��  ����@A�B33C55�                                    Bxo	@�  �          @�z����?\(�@�HA�{C*B�����L��@#�
A���C6T{                                    Bxo	Oz  �          @��H��?p��@�AθRC)0�����@&ffAޏ\C5��                                    Bxo	^   �          @�(���
=?��@��A�z�C'���
=�#�
@(Q�A���C4#�                                    Bxo	l�  �          @�ff��=q?���@33A��HC&����=q>��@%AظRC2L�                                    Bxo	{l  �          @�(���\)?�p�@33A���C&(���\)>8Q�@&ffA�=qC1�R                                    Bxo	�  T          @�z���G�?�p�@�A���C#����G�?�@p�A�33C.�                                    Bxo	��  �          @�����H?�?���A��\C!޸���H?B�\@�A�{C+�=                                    Bxo	�^  �          @������
?��
?�  Aw
=C �
���
?�  @ffA�33C(��                                    Bxo	�  �          @��R��@?���A7�C�q��?�Q�?�33A��C$z�                                    Bxo	Ī  �          @��
��33?+�@<��B�C+�R��33�(�@=p�BQ�C;��                                    Bxo	�P  �          @�����
?J=q@$z�A�Q�C*�q���
��{@*�HA�p�C8�                                    Bxo	��  �          @�=q��{?Tz�@�HA�  C*ff��{�u@"�\A��
C6�\                                    Bxo	�  �          @��\��z�?�G�@p�A�C(\)��z��G�@*=qA��C5:�                                    Bxo	�B  �          @�33���
?�=q?�Q�A�Q�C(!H���
>.{@p�A�ffC2�                                    Bxo
�  �          @��H��{?u?�\A�{C)�=��{>\)@ ��A���C2��                                    Bxo
�  �          @�(���  ?c�
@333A�Q�C)W
��  ��33@:=qA���C8B�                                    Bxo
+4  �          @��H����?��@(�A�p�C'h����ͼ�@*�HA��C4L�                                    Bxo
9�  �          @��\��p�?�{@{A�C$���p�>��R@%�A�G�C0ff                                    Bxo
H�  
�          @��H���H?�\)?���A�
=C"O\���H?Q�@
=A�C*�f                                    Bxo
W&  �          @������H<�@X��B�C3�
���H��33@EBG�CF�)                                    Bxo
e�  �          @�G��p�׾�
=@���B;�
C:h��p���@`  B�CQ�                                    Bxo
tr  �          @���g��+�@���BA�\C>� �g��ff@^�RB\)CU
=                                    Bxo
�  �          @�(��i����\)@�(�B={CE��i���0��@QG�BQ�CY!H                                    Bxo
��  �          @�p���\)@9��?333@�C����\)@�?�{A�33C�                                    Bxo
�d  �          @������@C�
>�=q@)��CǮ���@*=q?��Ax  CG�                                    Bxo
�
  �          @�Q���
=@0  ?J=q@�ffC
=��
=@	��?��A�\)C�\                                    Bxo
��  �          @�=q���@;�>���@S33C�)���@ ��?�ffAw\)CO\                                    Bxo
�V  �          @��H���@J=q>\)?�{C0����@3�
?�(�AiG�C@                                     Bxo
��  �          @��H��z�@H��>�  @�RCE��z�@.�R?ǮAy��C�                                    Bxo
�  �          @�����H@7�?�R@���CT{���H@�?��
A��HCE                                    Bxo
�H  �          @�����@)��>��@��C
=���@{?�  Au�C                                    Bxo�  T          @�ff��p�@-p�?fffAffC#���p�@�
?�p�A��RC=q                                    Bxo�  �          @�{��33@3�
?fffA�\C�R��33@	��@G�A��\C�                                    Bxo$:  �          @�����@8Q�?xQ�Ap�C������@�@
=A���Cs3                                    Bxo2�  �          @�����@6ff>�G�@�C�����@��?�\)A���C�H                                    BxoA�  �          @�p����@.{?@  @�=qC����@��?�A�  C�                                     BxoP,  �          @���G�@1�?�\)A6�\C�f��G�@�@p�A�C
                                    Bxo^�  �          @��R��ff@*=q?z�HA�RC�R��ff?�p�@�A�33C5�                                    Bxomx  �          @����\)@'�?�=qA.ffC=q��\)?�33@
=A���C&f                                    Bxo|  �          @�\)���R@+�?z�HA��C�
���R@   @�\A�G�C{                                    Bxo��  �          @�{��\)@#33?�  A!�C�f��\)?�\)@ ��A�33CxR                                    Bxo�j  �          @�p�����@p�?\(�A  C�3����?�?�A��C��                                    Bxo�  �          @�{���R@,��?
=q@�{Ck����R@��?�z�A�  C{                                    Bxo��  T          @�(���@Mp�����T  C{��@\��>u@=qC
                                    Bxo�\  �          @����z�@N�R��  �N{C����z�@\��>�\)@:�HC�
                                    Bxo�  �          @�  ��p�@4z��Q����C���p�@^{�0����C@                                     Bxo�  �          @�G����\@Fff��p��x��CW
���\@\(�<��
>k�C��                                    Bxo�N  �          @������@O\)���H�s�Cٚ����@c33=���?�=qCJ=                                    Bxo��  �          @�33��G�@Tz΅{�`z�CG���G�@e�>u@=qC.                                    Bxo�  �          @��H����@Tz῱��f�RC!H����@e>W
=@�C��                                    Bxo@  �          @������@G�������{C�����@`  ��\)�333C�
                                    Bxo+�  �          @������H@A녿�z���\)C)���H@\�;#�
��33C��                                    Bxo:�  �          @��\���
@B�\�8Q���ffC�����
@B�\?0��@�33C��                                    BxoI2  �          @�33����@>�R�h����CQ�����@E�?   @�z�C�                                     BxoW�  �          @�����@:�H���R�MCW
���@J�H>8Q�?��C5�                                    Bxof~  �          @�=q��  @<(���\)�J=qC����  @*�H?�p�AT  C^�                                    Bxou$  �          @��H���@?\)?�33Ah  C\)���@�
@%�A�(�CW
                                    Bxo��  T          @����z�@@  >�@��RC
��z�@�R?�G�A�{C��                                    Bxo�p  �          @�(���  @8Q�?W
=A
=qC�q��  @p�@�\A�  C\                                    Bxo�  �          @�����G�@?\)?�ffA/\)C����G�@p�@G�A��
C�                                    Bxo��  �          @�������@=p�?��A=��C������@	��@A�  C�H                                    Bxo�b  �          @�33����@>{?n{A��Cu�����@\)@	��A�33CB�                                    Bxo�  �          @�����z�@?\)?�@���C=q��z�@(�?�A�z�CT{                                    Bxoۮ  �          @�=q���@I��?�\@�\)CY����@%?�{A�ffCc�                                    Bxo�T  �          @������@J=q>�{@aG�C33����@+�?�(�A�z�C�=                                    Bxo��  �          @�=q��=q@J�H=�?��RC33��=q@1�?��
A�C��                                    Bxo�  �          @����p�@?\)�u�!G�C\)��p�@1�?�\)A;
=C5�                                    BxoF  �          @�\)��  @G�����*�HC8R��  @9��?�
=AF=qC#�                                    Bxo$�  �          @�
=��  @C�
��ff����C���  @<(�?xQ�A$  Cٚ                                    Bxo3�  T          @�\)��{@Fff�J=q���C���{@HQ�?0��@�G�C�
                                    BxoB8  �          @�=q���@A녿�  �$��Cc����@I��>�@�{CT{                                    BxoP�  �          @����  @E��G��P(�C����  @U�>�=q@1G�C�\                                    Bxo_�  �          @�33���@4zῨ���Y�C�{���@G
==���?�G�CG�                                    Bxon*  �          @��\���\@-p���Q����\Cc����\@K���=q�4z�C:�                                    Bxo|�  �          @����G�@0�׿У���z�C�f��G�@L�;L����
CǮ                                    Bxo�v  �          @�33����@4z��z����\C{����@QG��L����
C+�                                    Bxo�  �          @��H��G�@E���\�(��CǮ��G�@Mp�>��H@��\C�3                                    Bxo��  �          @��H��\)@I����{�8(�C����\)@S�
>�G�@���C�                                    Bxo�h  �          @��H����@HQ�xQ�� (�C\)����@N{?\)@���C��                                    Bxo�  �          @��H��ff@S33�8Q���ffC� ��ff@P��?\(�AC�                                    BxoԴ  T          @�33����@N{����z�C}q����@HQ�?uA33CL�                                    Bxo�Z  �          @�(����@XQ쿊=q�0z�C5����@`  ?
=@���CB�                                    Bxo�   �          @�(�����@QG���G��N�RC\)����@^�R>\@x��C��                                    Bxo �  �          @����{@`  �����F�HC���{@j�H?
=q@���C
�{                                    BxoL  �          @�(���{@`  ����T  C.��{@l��>�@�  C
�)                                    Bxo�  �          @��\����@QG���G���z�C������@n{�u�#�
C	0�                                    Bxo,�  �          @��
���\@1G��+���=qC�����\@l(������5G�C	�{                                    Bxo;>  �          @�z����R@>�R����
=C�q���R@i���\)��\)C�                                    BxoI�  �          @�(��u@P��� �����C�f�u@����8Q���Cs3                                    BxoX�  �          @���@.�R�&ff���C����@hQ쿅��*�\C�                                    Bxog0  �          @����\)@S�
��p����HC����\)@n�R    ���
C
�\                                    Bxou�  �          @�����=q@k�����S�
C	�H��=q@vff?\)@��RC��                                    Bxo�|  �          @�z���@c33����T  C����@p  ?   @�z�C
)                                    Bxo�"  �          @������
@O\)�����c�Cn���
@`��>�\)@:=qC5�                                    Bxo��  �          @�z���33@b�\���
�P��Cٚ��33@R�\?�{A`��C�3                                    Bxo�n  �          @�������@b�\���
�uC33����@H��?У�A��CxR                                    Bxo�  �          @�����\)@\(�>��?��Cz���\)@>{?޸RA�G�Cn                                    Bxoͺ  �          @�ff��\)@`  >L��?�(�C���\)@@  ?�A�  C.                                    Bxo�`  �          @�
=��33@XQ�=���?��C�q��33@<(�?�A�  Cu�                                    Bxo�  �          @�z���(�@`  �   ����CY���(�@U�?���AD��C�                                    Bxo��  �          @����\)@e���
�(��C����\)@j=q?E�@�33C)                                    BxoR  �          @��
����@G
=�����EC5�����@8��?�Q�ADz�C!H                                    Bxo�  �          @�33��z�@Dz�
=��G�C���z�@>�R?k�A�C=q                                    Bxo%�  �          @���{@G��333��(�CO\��{@E?Y��A	G�C�)                                    Bxo4D  �          @�ff���
@HQ쿎{�3�
C�����
@Q�>��H@�
=C��                                    BxoB�  �          @�p����H@QG�������\)C����H@h��>�?��C�                                    BxoQ�  �          @��
����@U�����o�
C�����@hQ�>���@A�C�)                                    Bxo`6  �          @��
��=q@Z�H�����/\)C����=q@aG�?+�@�z�Cٚ                                    Bxon�  �          @�(���p�@P�׿����D��C�{��p�@\(�>��H@�ffC�                                    Bxo}�  �          @������R@@�׿(����{Ck����R@=p�?Y��A
ffC�
                                    Bxo�(  �          @����Q�@<(��
=��G�CJ=��Q�@7�?^�RA�\C�f                                    Bxo��  �          @�����z�@Fff�����z�C8R��z�@AG�?p��A�RC��                                    Bxo�t  �          @��H��p�@S33�fff��CJ=��p�@U�?G�@��C
=                                    Bxo�  �          @��
���\@Tz῏\)�9��C����\@]p�?
=@�33Ck�                                    Bxo��  �          @��
�AG�@Mp��W���
CE�AG�@���Q��q�B���                                    Bxo�f  �          @�33�"�\@O\)�q��*(�C 
�"�\@��Ϳ��
��z�B�L�                                    Bxo�  �          @�33��H@G��}p��4G�B�����H@�(����R����B�33                                    Bxo�  �          @�z��   @.{��
=�S��B����   @����(Q��߮B�3                                    BxoX  �          @�p��:=q@j�H�G���RC k��:=q@��R�}p��!B�\)                                    Bxo�  A          @������@n{�]p��p�B����@���G��N�\B��                                    Bxo�  g          @�z���\@^{�{��1\)B����\@������z�B���                                    Bxo-J  �          @��,(�@q��QG��z�B���,(�@�zῆff�+
=B�                                    Bxo;�  �          @����   @|���Fff�p�B��{�   @�ff�Tz���
B�z�                                    BxoJ�  �          @���)��@i���Z�H��RB����)��@��H��  �K�B�W
                                    BxoY<  �          @����Dz�@vff�/\)���HC ���Dz�@�ff������B�G�                                    Bxog�  �          @���XQ�@z�H�{��(�C�XQ�@�  �u�&ffB��R                                    Bxov�  �          @����Z�H@}p��Q���G�C�
�Z�H@��=#�
>�
=B��\                                    Bxo�.  �          @��j=q@c33�   ���HC�{�j=q@�=q�����G�CB�                                    Bxo��  �          @�ff�^{@j�H�(����
=CaH�^{@���
=q��ffB�\)                                    Bxo�z  �          @�p��h��@b�\�p����CǮ�h��@�G���ff����CJ=                                    Bxo�   T          @�z��k�@Y���$z���33C	J=�k�@�
=�������C�                                    Bxo��  �          @�z��fff@U��0����
=C	G��fff@�  �L���\)CL�                                    Bxo�l  �          @���`��@N�R�A���
C	L��`��@�����=q�0��C.                                    Bxo�  �          @��i��@U�2�\��33C	�=�i��@��׿Q��(�C�                                     Bxo�  �          @�ff�^{@^�R�1G���C�f�^{@�z�:�H��\)C O\                                    Bxo�^  �          @�
=��@���)�����HB��)��@�
=�����
B�33                                    Bxo	  �          @�ff�%�@�\)�3�
��Q�B��)�%�@�녾Ǯ�~{B��                                    Bxo�  �          @�{�N�R@�
=��(���Q�B��H�N�R@���>�{@]p�B��                                    Bxo&P  �          @��j=q@l���
=q��(�C���j=q@��ý��
�O\)C��                                    Bxo4�  �          @�{�[�@o\)�"�\��z�C� �[�@�Q�Ǯ�z=qB��\                                    BxoC�  �          @�{�^�R@q�����ffC���^�R@����=q�.{B��{                                    BxoRB  �          @��l��@mp��
=q��p�C���l��@�����\)�5C�R                                    Bxo`�  �          @�p��g
=@u��ff���
CJ=�g
=@��
=L��>�C��                                    Bxoo�  T          @��z=q@i����=q����C��z=q@��\>L��@G�CǮ                                    Bxo~4  �          @����l��@A��;����
C�R�l��@�=q����/33C@                                     Bxo��  T          @��
���R@:=q��
�£�CW
���R@j�H�
=���C
�f                                    Bxo��  �          @����u�@N{�&ff�ۙ�C�R�u�@��\�.{��C33                                    Bxo�&  �          @�=q�C33@;��aG��  C0��C33@��ÿ�\)���B�                                      Bxo��  �          @�  �1�@Mp��^�R�G�C�3�1�@�Q쿹���t��B�k�                                    Bxo�r  �          @�
=�(�@=p��w
=�4��C�\�(�@�
=��33����B�=q                                    Bxo�  �          @��H��
@e�o\)�(\)B����
@�
=��p��u�B�                                      Bxo�  �          @�����@]p��{��4B�.��@�ff���H���B�#�                                    Bxo�d  �          @�G�� ��@k��S33��B��{� ��@��\��ff�.ffB��f                                    Bxo
  �          @����  @E�x���5�HB�\�  @��������z�B�                                      Bxo�  �          @�p��XQ�@=q�_\)� =qC�{�XQ�@tz������C�=                                    BxoV  �          @��
�Y��@p��a��$33C��Y��@j�H��(����C�=                                    Bxo-�  �          @��H�`  @(Q쿂�\�Qp�C
=�`  @2�\>�
=@�z�Ck�                                    Bxo<�  �          @�=q� ��@*�H@HQ�B,Q�B�� ��?B�\@�G�BwC33                                    BxoKH  
(          @���Z�H@AG�@�\A��HC
�
�Z�H?�{@QG�B#�\C��                                    BxoY�  �          @�ff��Q�@`�׿L�����C
Ǯ��Q�@\(�?�ffA4Q�CT{                                    Bxoh�  �          @�p���Q�@Z=q���
�1��C����Q�@^{?L��A	C�                                    Bxow:  �          @�����ff@`  �����RC+���ff@Tz�?�  ARffC��                                    Bxo��  �          @��R����@>{��(�����Cn����@c�
�aG��Cn                                    Bxo��  �          @�{���@-p��\)����C����@^{�
=��  C+�                                    Bxo�,  �          @�
=���@#33��(�����C�����@L(���G����HC{                                    Bxo��  T          @����?��H��(����C����@,(��:�H��Q�C�                                    Bxo�x  �          @�p����?��R����
=C����@;���{�<��CaH                                    Bxo�  �          @�{��p�@\)�\)�\C.��p�@C�
�Q��
ffCY�                                    Bxo��  �          @��
��\)@  �ff��
=C�R��\)@HQ�fff�  C��                                    Bxo�j  �          @��|(�@H�ÿ�33����Cu��|(�@j�H��\)�B�\C	�                                    Bxo�  �          @�p���G�@���(��׮C���G�@Dzῃ�
�1p�Cc�                                    Bxo	�  T          @���z�@0���
�H��z�CB���z�@^�R���H��z�C�                                    Bxo\  �          @�(�����@$z������z�C&f����@L�;�����\)CQ�                                    Bxo'  �          @�33����@z��p��ƣ�CO\����@9���^�R��C�                                    Bxo5�  �          @�33�~�R?���L(���C�f�~�R@;���p���Q�C�H                                    BxoDN  �          @���%?!G���G��n=qC&k��%@5��fff�,C}q                                    BxoR�  �          @����>�\)��G��RC,
��@*=q��  �H(�B���                                    Bxoa�  �          @����Mp�?8Q��333�!��C'L��Mp�@�\�33����C��                                    Bxop@  �          @�
=����@K�?��\A,��C������@�R@\)Aٙ�C�)                                    Bxo~�  �          @�
=��
=@E?�=qAd��C��
=?��R@.{A�C�R                                    Bxo��  T          @�����G�@8��@�RA�33Cn��G�?�\)@X��B�RC!Y�                                    Bxo�2  �          @�=q���@'�@'
=A�RC&f���?p��@e�B"�C&��                                    Bxo��  �          @����Q�@6ff<��
>W
=C���Q�@��?��HAt��CǮ                                    Bxo�~  T          @�����(�@p���{���C5���(�@6ff����\)C��                                    Bxo�$  �          @��aG�?�Q����H�9Q�CT{�aG�@b�\�*�H���HC޸                                    Bxo��  �          @�{��
=@��<(���C���
=@a녿����X  C�                                    Bxo�p  �          @�
=���\@
=q�U�Q�C)���\@c�
��\����C
�                                    Bxo�  �          @�
=���H@(���'���(�C����H@g
=�h���33C
�{                                    Bxo�  �          @�Q���z�@]p����
�N�RC�3��z�@hQ�?(��@�(�C\)                                    Bxob  �          @�Q��g�@XQ��:=q��p�C��g�@�z�G���G�Cz�                                    Bxo   �          @�Q��dz�@Y���>�R��p�Cc��dz�@�{�Tz����C                                     Bxo.�  �          @�  �a�@Z=q�=p�����C���a�@�{�L��� ��C p�                                    Bxo=T  �          @����\)@_\)�����(�C
�=�\)@�
=��=q�*=qC\)                                    BxoK�  �          @�����  @c�
��{���C���  @xQ�>�(�@�G�C	��                                    BxoZ�  �          @�  ���@mp���33���
C	�����@�G�>�@���C0�                                    BxoiF  T          @��H����@E���R�ə�CO\����@z=q�\)��\)C	�{                                    Bxow�  �          @��H���
@A��333��\)C�f���
@�G��Y����C�{                                    Bxo��  �          @����c�
@%��l��� G�C��c�
@��������ffC�H                                    Bxo�8  �          @�ff�8��@ ����
=�?  C
���8��@�=q������B��                                     Bxo��  �          @�\)�G�@1��xQ��+��C
\)�G�@�(���z���(�B�                                    Bxo��  �          @�Q��tz�@(��^�R�G�Ch��tz�@xQ�޸R��p�C�                                     Bxo�*  �          @�  �qG�@(��l���!33C���qG�@q���\���C��                                    Bxo��  �          @���p  @�H�dz���RC+��p  @z=q������Q�CǮ                                    Bxo�v  �          @�
=�\)@$z��G��G�C.�\)@s�
�����\  CL�                                    Bxo�  �          @�Q��E@S33�dz��=qC
=�E@�p���{�Z{B��H                                    Bxo��  �          @����  @ff�Y����HC}q��  @qG��ٙ���G�C��                                    Bxo
h  �          @�\)���H@1G��9����
=C�H���H@w���ff�)�C��                                    Bxo  T          @�����=q@p��:=q����CW
��=q@g
=��(��D  C�                                    Bxo'�  �          @������@�H�9����G�C�R���@dz῝p��E�C�3                                    Bxo6Z  �          @�  ��ff?�(��B�\����C
��ff@P�׿�=q��C�=                                    BxoE   �          @�
=��{?�  �Q��{C!O\��{@>�R����z�C
                                    BxoS�  �          @�ff��(�@Q��I�����C���(�@j�H���H�mp�C
k�                                    BxobL  �          @����Q�@3�
�>{��C���Q�@|�Ϳ�=q�.{Cu�                                    Bxop�  �          @�G��xQ�@C33�>{��Q�C�{�xQ�@��Ϳu�G�C
                                    Bxo�  �          @����x��@P  �2�\����C
�x��@�\)�333��
=C�
                                    Bxo�>  �          @�=q���
@/\)�G����\C\)���
@XQ쾙���A�C�                                    Bxo��  �          @�����
@*�H�G����RC���
@]p��\)����C.                                    Bxo��  �          @��H��=q@$z�� ����C����=q@O\)�\�r�\C�                                    Bxo�0  �          @�����  @6ff��R���C����  @fff��
=��ffCW
                                    Bxo��  �          @�=q���@&ff�=q���
C�
���@^�R�5��
=C                                      Bxo�|  �          @�����33@.�R�������CB���33@^�R��ff��Q�C�H                                    Bxo�"            @�  ���R@"�\�%��{C^����R@`�׿c�
�{CǮ                                    Bxo��  �          @�����(�@(Q��,(���{C���(�@i���k��\)C8R                                    Bxon  �          @�G���{@ff�<������C����{@U����e��C.                                    Bxo  �          @������@'��0  ��33C  ���@j�H�z�H�  C��                                    Bxo �  �          @�\)���R@���AG���CO\���R@g���=q�V�HC@                                     Bxo/`  �          @������\@{�7����CW
���\@fff��z��9G�CE                                    Bxo>  �          @�ff�&ff@n{�W
=���B��q�&ff@��k��p�B�                                    BxoL�  �          @��R�5@Tz��g���RC� �5@�\)�����\z�B��                                    Bxo[R  �          @��Q�@fff�tz��)B�33�Q�@�=q��\)�`��B�{                                    Bxoi�  �          @�\)��
=@{��s33�&Q�B�Q��
=@��H���=��B���                                    Bxox�  �          @�
=��  @u�|���/G�Bޮ��  @��H��{�\(�B�\)                                    Bxo�D  �          @�\)��G�@k����H�E��B�� ��G�@����\��(�B�33                                    Bxo��  �          @�ff�G�@s33����<  B�.�G�@�p���ff�|��B�aH                                    Bxo��  T          @�\)���H@g
=��Q��@�HB�(����H@�����p���G�Bͽq                                    Bxo�6  �          @�
=����@P�����R�NB��ÿ���@�����Q�Bъ=                                    Bxo��  �          @�
=���R@U����
�Hp�B�8R���R@�(���(����B�k�                                    BxoЂ  �          @���{@A��xQ��7�HB�� �{@�����H���B�\)                                    Bxo�(  �          @�p��p  @I���%���ffC��p  @�G��
=q���HC޸                                    Bxo��  �          @��R���\@=p��\)���RC�����\@l(���33�^�RC�\                                    Bxo�t  �          @������@7
=�(����CaH����@mp�����{C33                                    Bxo  �          @�\)��Q�@/\)�{���HC���Q�@`  ��ff��
=C&f                                    Bxo�  T          @�����@-p���(�����C�����@U��  �{C\)                                    Bxo(f  �          @�  ����@3�
�(����HC)����@b�\�\�q�C�f                                    Bxo7  �          @����z�@S�
�
=q���CO\��z�@}p���Q�fffCJ=                                    BxoE�  �          @�Q�����@O\)�ff��
=C�H����@w
=���
�@  C	�3                                    BxoTX  �          @�����z�@ff�33����C� ��z�@@  �W
=�C}q                                    Bxob�  f          @������\@Dz��  ��C�����\@s33��z��7
=C
��                                    Bxoq�  �          @��H���@N{�"�\��=qC=q���@�=q����
=C��                                    Bxo�J  �          @��\�u@U�6ff��Q�C
�q�u@��H�(����\)Cz�                                    Bxo��  �          @����}p�@Mp��0����C��}p�@��#�
��z�Cff                                    Bxo��  �          @�(��c�
@8Q��L���C���c�
@�zῘQ��D��C�                                    Bxo�<  �          @����=q@:=q��R��ffCn��=q@i����33�mp�C
�                                    Bxo��  �          @����^{@0  �E���\C���^{@}p���z��G33C0�                                    BxoɈ  �          @�(���G�?������tffC{��G�@���=p��  B�Ǯ                                    Bxo�.  �          @�(��(�?����z�.B�=q�(�@y���g��*=qBŮ                                    Bxo��  �          @�p��dz�@Tz��(���\)C	
�dz�@~{���Ϳ��
C�                                    Bxo�z  �          @�����  @a녿�
=�p��C
����  @p  ?(��@ڏ\CǮ                                    Bxo   �          @��\�j=q@K��*�H���C�j=q@���z����RC�                                    Bxo�  �          @����@��@_\)�@�����C���@��@�녿333����B���                                    Bxo!l  �          @�ff�*=q?�ff��G��T�\C�3�*=q@s33�+���B�                                    Bxo0  �          @�Q��(�@P  �<(����B���(�@�=q�E��(�B�Ǯ                                    Bxo>�  �          @�G���G�    ��{k�C4(���G�@'���{�\�
B�B�                                    BxoM^  �          @�p���\)�ٙ����Rk�Cl�{��\)?�����33W
C
=                                    Bxo\  �          @������{���R�p{CuJ=���>�
=���L�C�                                    Bxoj�  �          @�ff���R�
�H��Q��y\)CpQ쿞�R?333��aHC��                                    BxoyP  �          @�Q�n{�Dz���ff�X�HC}��n{�k������
CAǮ                                    Bxo��  �          @�=q���	�������=C����?J=q��G� 33B�                                      Bxo��  �          @�(��O\)��Q���{\)Ck�O\)?������B�ff                                    Bxo�B  �          @��
��(���z���
=��Ci�H��(�?�\)�����RC�R                                    Bxo��  �          @�p����\��(�����(�Cm=q���\?�=q���Cz�                                    Bxo  �          @��R��\)��=q��Q���Cnh���\)?�G���p�k�C��                                    Bxo�4  �          @�\)���1G����
�W�Ck.��>���(���C0=q                                    Bxo��  �          @��׿�\)�������C`����\)?\��{C)                                    Bxo�  �          @�Q��R�5��¡8RCd�׿�R@���L�BЀ                                     Bxo�&  �          @�
=�'��(����H�I�RC^�3�'�>�  ��
=�xG�C.��                                    Bxo�  �          @���7
=�
�H����M=qCY&f�7
=?\)��ff�m�C(��                                    Bxor  �          @���'���z���
=�p(�CK�
�'�?�
=��=q�d��Cc�                                    Bxo)  �          @�
=�L(�>������[�RC1�)�L(�@"�\�qG��+
=Cp�                                    Bxo7�  �          @����>{�Ǯ����jffC;xR�>{@�\����E  CO\                                    BxoFd  �          @�Q��:=q��z���ff�m�C9��:=q@�����\�C�HCu�                                    BxoU
  �          @����Dz�+����
�dC@G��Dz�@�\��
=�J
=C\)                                    Bxoc�  S          @����G
=�B�\���\�a��CA�R�G
=?�Q�����J�C                                    BxorV  �          @�G��hQ�>�Q���
=�K(�C.c��hQ�@,(��e���Ck�                                    Bxo��  T          @����Z=q�\)�����UC=h��Z=q@G���\)�;�RCaH                                    Bxo��  �          @��H�U������Q��Z�C:�=�U@p���\)�:33Cz�                                    Bxo�H  �          @��H�Mp���������Z�CFǮ�Mp�?������P�C��                                    Bxo��  �          @����U��s33����S33CC޸�U�?�33����F=qC��                                    Bxo��  �          @�  �W��\)����U��C=k��W�@   ��{�;��CJ=                                    Bxo�:  �          @�{�XQ�h����\)�P33CC\�XQ�?�������B��C!H                                    Bxo��  �          @��R�Tzῢ�\��ff�M�RCH���Tz�?�����{�L�
CW
                                    Bxo�  T          @����U��\)��G��G��CJff�U?������L�C!�                                    Bxo�,  �          @�33�'
=�@  ���H�r�RCC�3�'
=?�(���\)�W��C
=                                    Bxo�  �          @�(���33?����z�B�B�W
��33@�33�U����B�L�                                    Bxox  
�          @���u@?\)��G��g�
B�G��u@�33�����B�W
                                    Bxo"  �          @�ff�h��@2�\����j�\B�#׿h��@��#�
��Q�B��H                                    Bxo0�  T          @��
?�=q@'���  �l��B�33?�=q@�Q��&ff��ffB�u�                                    Bxo?j  T          @��
���?�Q����R�B�����@|(��~{�4\)B��f                                    BxoN  �          @��\�(�?�33������B�W
�(�@����n{�)p�B�W
                                    Bxo\�  �          @�녿��R�@���x���D�
Cs�H���R��ff�����qCD�H                                    Bxok\  �          @��H�G
=��  �����Mp�Ci�G
=�Z�H�!���Cc��                                    Bxoz  T          @��H�)����=q>��?�33Co��)���|(�����G�Cl�                                    Bxo��  �          @�(����
��\)�333�Q�CFff���
>�33�Fff�ffC/!H                                    Bxo�N  �          @�����{���
�Y���=qCHٚ��{��ff��
=��(�C@�
                                    Bxo��  �          @�p��|(��*=q?�z�A�(�CU���|(��HQ������CZz�                                    Bxo��  T          @�z����H�:=q?�p�AS�
CU�
���H�G������CW��                                    Bxo�@  �          @������)��?��RA�ffCR�f���S33>�  @$z�CX�R                                    Bxo��  T          @�=q������@z�A��
CQ#�����H��>�ff@��CX�                                    Bxo��  �          @��������HQ�>�
=@��CV�H�����7
=��=q�^ffCTO\                                    Bxo�2  �          @������\�,(�>��@,(�CQ&f���\�=q��p��M��CN��                                    Bxo��  �          @��R��ff�=q��\)����CPu���ff��z��*=q��ffCB��                                    Bxo~  �          @�p��~{���J�H��\CL�)�~{>#�
�j=q�*�\C1                                    Bxo$  �          @����n{�Mp��
=���C\���n{�����`  �#�RCJ�\                                    Bxo)�  �          @����B�\�~{�   ��=qCh���B�\�G��tz��4�HCX��                                    Bxo8p  �          @�z��4z������
�H��{Ck��4z���R��  �@(�CZQ�                                    BxoG  �          @�p�����z��.{����Cr�����G������dQ�C_��                                    BxoU�  �          @����ff�mp��b�\�"��Ct���ff���H��\)�fCV                                      Bxodb  �          @����(Q��K��h���%��Cfu��(Q�8Q������qG�CCL�                                    Bxos  �          @�\)�B�\�:�H�Z�H�33C_ٚ�B�\��R���R�\��C?n                                    Bxo��  �          @��R�Tz��3�
�P  �{C\8R�Tz�(���  �N(�C>s3                                    Bxo�T  �          @���H�ÿ�(��u��B�CI&f�H��?���x���F(�C!��                                    Bxo��  �          @��B�\���
��z��MCJ��B�\?�z���p��P33C+�                                    Bxo��  T          @�\)�Q�?�33����D=qC��Q�@X���-p����C                                    Bxo�F  �          @�  �g�@��XQ��\)C���g�@o\)��\)��(�C
=                                    Bxo��  �          @��H�r�\@���S33���C33�r�\@k���ff���Cٚ                                    Bxoْ  �          @���N{@e�8Q����RCٚ�N{@��\���H���RB�33                                    Bxo�8  �          @�
=���
@��A���C�����
@[���z��k�C=q                                    Bxo��  �          @�{���R@�(����\)C����R@X�ÿs33� (�C#�                                    Bxo�  �          @��
�-p�@O\)�l(��$�C�)�-p�@�
=�����_�B�                                    Bxo*  �          @����mp�@7
=�G��z�Cn�mp�@��H��=q�1�CB�                                    Bxo"�  �          @�  ����?�p��:�H��
=CO\����@P  ����`Q�CO\                                    Bxo1v  �          @�����z�?����
=q����C$�q��z�@���=q�,��C�H                                    Bxo@  �          @�����z�?��\�
=��  C&!H��z�@��ff�O33C�                                    BxoN�  �          @��\��z�?k��%�ң�C)ٚ��z�@
�H��
=���C)                                    Bxo]h  T          @������\?k��#33��z�C)����\@	����33��p�C�                                    Bxol  T          @����(�>�{�*=q���HC0B���(�?�(���
���\C!xR                                    Bxoz�  �          @������
?@  �%�ԏ\C+�R���
@G����
����Ck�                                    Bxo�Z  �          @��
����?#�
��R��{C-\����?�׿޸R��{C c�                                    Bxo�   �          @�(���(�?8Q��/\)�ޣ�C,
=��(�@���z����\C�                                    Bxo��  �          @��\��z�?z��C�
���\C-33��z�@�������\)C\)                                    Bxo�L  �          @�=q��(�?����Z=q�(�C%�
��(�@2�\�����ffC�                                     Bxo��  �          @�\)��G�?�{�AG��z�C!aH��G�@1녿���=qCxR                                    BxoҘ  �          @�Q���  @Z�H�\)���\C� ��  @��H�#�
����Cff                                    Bxo�>  �          @�\)��33@fff�������\C
���33@xQ�?�R@ə�C�\                                    Bxo��  �          @������@AG���z���{C8R����@\(�>k�@ffC�3                                    Bxo��  �          @�  ��@��/\)���C�\��@^�R���\�$��C޸                                    Bxo0  �          @�\)��33@��4z���
=Cc���33@aG�����0z�C�                                    Bxo�  �          @���[�?Ǯ�����>��C���[�@c�
�)���ᙚC�                                    Bxo*|  T          @���W�@�  ?У�A�ffC.�W�@�R@a�B p�C�f                                    Bxo9"  �          @��
�j�H@�{>�Q�@j=qC0��j�H@U�@#�
AڸRC	�R                                    BxoG�  �          @�33�tz�@\)>��H@��C���tz�@E@$z�AܸRC�3                                    BxoVn  �          @�z��HQ�@��\?fffAB����HQ�@J=q@FffB��C��                                    Bxoe  �          @�z��?\)@��?uA��B�z��?\)@Y��@Tz�B�HCW
                                    Bxos�  �          @�p���Q�@|��>aG�@\)Cn��Q�@L��@z�A�ffC\)                                    Bxo�`  �          @�p��|(�@��=��
?J=qC(��|(�@W�@��A�(�Cp�                                    Bxo�  �          @�����@QG��O\)��
C&f����@L(�?��A1p�C�{                                    Bxo��  �          @�����@G������\)Ch����@#33�k��z�Cz�                                    Bxo�R  �          @�G����?�z��z���=qC'�����@��{�2ffCB�                                    Bxo��  �          @�  ��\)?n{������33C)����\)?�����9p�C ��                                    Bxo˞  �          @�  ��G�?h���G���33C*J=��G�?��Ϳ��H�C�C ��                                    Bxo�D  �          @�  ���R?+���=q��=qC-����R?��xQ��  C%u�                                    Bxo��  �          @������>�Q��(�����C0@ ���?�p���  �H��C'^�                                    Bxo��  �          @�33����?
=��(����C-޸����?�
=�����3\)C%}q                                    Bxo6  �          @�(���p��=p���{�0z�C;h���p����
����T(�C40�                                    Bxo�  �          @�������>�����\z�C/�����?�z�h���\)C(k�                                    Bxo#�  �          @�z���G�?Ǯ�����<��C$B���G�?��H�.{��z�C z�                                    Bxo2(  �          @����  @��333�޸RC�3��  @�?0��@ٙ�C��                                    Bxo@�  �          @��R�y��@��
��\)�.{Cs3�y��@`��@
�HA��C
�                                    BxoOt  �          @���U@���<#�
>��B���U@s33@��A���CW
                                    Bxo^  �          @��
�B�\@�p��L���33B���B�\@�G�@ffAǅB��f                                    Bxol�  
k          @�(��E�@��R<�>���B�p��E�@|��@$z�A�
=C                                       Bxo{f  �          @�z��=q@��\<�>�=qB�3�=q@���@0��A�p�B��f                                    Bxo�  �          @�p���R@�=q�u���B�.��R@���@!�A���B��)                                    Bxo��  
�          @�
=��\)@s33>k�@ffC

��\)@Dz�@\)A��
C�                                    Bxo�X  
(          @�����{@x�ý#�
�\C	!H��{@S33@�
A�\)C�\                                    Bxo��  
�          @������@��>��R@S33CǮ���?��?�Aq��C!�)                                    BxoĤ  T          @�����ff��G�>���@U�CEٚ��ff��z�+����CD�                                    Bxo�J  �          @����녾\?˅A33C7�f��녿�
=?��A5�C?��                                    Bxo��  
�          @�������@Q�A�\)C5�������\?�(�A���CA.                                    Bxo�  
(          @�G�����?��?�z�A���C"������>�@�A��
C.Ǯ                                    Bxo�<  
�          @�Q���=q@.�R?���A3\)C����=q?��H@��Aȏ\C ��                                    Bxo�  
�          @�33�z=q@{�?(��@�G�C�=�z=q@=p�@*�HA���Cٚ                                    Bxo�  "          @��ÿ�\)@�  �+����B�#׿�\)@�G�@��A�p�B��
                                    Bxo+.  �          @��ÿ�  @��׾����{BӮ��  @�ff@(�Aҏ\B�Q�                                    Bxo9�  
�          @��\��@�(�>B�\?�p�B���@�Q�@1�A��B�                                     BxoHz  
�          @����7�@��R������B���7�@�Q�@�A��HB��                                    BxoW   T          @�=q�=q@�{��(���  B�{�=q@�z�@�\A�z�B�=                                    Bxoe�  �          @����QG�@�ff���Ϳ�  B����QG�@s33@�
A��C��                                    Bxotl  T          @���AG�@�G�?#�
@�B�\)�AG�@^�R@>�RBp�C�R                                    Bxo�  "          @���*�H@�G�?�p�A�G�B��*�H@9��@y��B2��C��                                    Bxo��  �          @�G��1�@�
=?ٙ�A�p�B��1�@7
=@uB/��C33                                    Bxo�^  
�          @�=q����@�p�?ǮA��RB�����@aG�@�33B=�HB�G�                                    Bxo�  �          @��H��(�@�\)@�\A���Bϔ{��(�@Fff@�z�BSG�B���                                    Bxo��  �          @����$z�@�p�?�33A�G�B�\)�$z�@-p�@~�RB;\)C��                                    Bxo�P  
�          @���~�R@{�?L��A�\Ck��~�R@8Q�@1�A�Q�C!H                                    Bxo��  T          @����x��@z�H?�\)A7�C���x��@,��@C33B
=C:�                                    Bxo�  
�          @����p��@z=q?\A{
=C���p��@�R@XQ�B��C�
                                    Bxo�B  
�          @�{�xQ�@u?���A�=qCQ��xQ�@Q�@Z=qB�Cs3                                    Bxo �  T          @�\)�vff@�  ?��AaG�C��vff@(Q�@Tz�B�
C��                                    Bxo �  	`          @�Q��q�@�p�?�A;�
C.�q�@8��@N�RBz�C��                                    Bxo $4  
�          @��p��@�ff?\)@�
=C�f�p��@O\)@.�RA�(�CJ=                                    Bxo 2�  
�          @���q�@�?�R@��C��q�@L(�@0��A�C��                                    